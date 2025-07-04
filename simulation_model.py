import torch
from transformers import BertModel, BertTokenizer
from torchvision import models
import torch.nn as nn




class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.backbone = models.efficientnet_b4(pretrained=True)
        self.backbone.classifier = nn.Identity()  # Remove final classification layer, because we're using this as an encoder

    def forward(self, x):
        # x's shape: (batch_size, 8, 3, 224, 224)
        print(x.shape)
        x = x.squeeze(2) 
        print(x.shape)
        batch_size, num_views, c, h, w = x.shape # num_views = 8 
        x = x.view(batch_size * num_views, c, h, w)
        features = self.backbone(x)  # (batch_size * num_views, feature_dim)
        features = features.view(batch_size, num_views, -1)  # Reshape back
        return features 
        # y's shape: (batch_size, num_views, feature_dim)
    


class TextEncoder(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super(TextEncoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)

    def forward(self, text):
        # Assuming text is a list of 2 strings (a sentence pair)
        if isinstance(text, list) and len(text) == 2:
            # Handle sentence pair for BERT (text, text_pair)
            inputs = self.tokenizer(text[0], text[1], return_tensors="pt", padding=True, truncation=True)
        else:

            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        outputs = self.bert(**inputs)
        pooled_output = outputs.last_hidden_state.mean(dim=1)  
        return pooled_output
    

class MultiModalFusion(nn.Module):
    def __init__(self, image_feat_dim, text_feat_dim, spawn_feat_dim, hidden_dim):
        super(MultiModalFusion, self).__init__()
        self.image_proj = nn.Linear(image_feat_dim, hidden_dim)
        self.text_proj = nn.Linear(text_feat_dim, hidden_dim)
        self.spawn_proj = nn.Linear(6, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=8)

    def forward(self, image_features, text_features, spawn_positions):
        image_features = self.image_proj(image_features)
        text_features = self.text_proj(text_features)
        spawn_features = self.spawn_proj(spawn_positions)

        combined_features = text_features + spawn_features
        image_features = image_features.permute(1, 0, 2)
        combined_features = combined_features.unsqueeze(0)

        fused_features, _ = self.cross_attention(
            query=image_features,
            key=combined_features,
            value=combined_features
        )

        fused_features = fused_features.permute(1, 0, 2)
        return fused_features.mean(dim=1)


        # Reshape back and aggregate across views
        fused_features = fused_features.permute(1, 0, 2)  # (batch_size, num_views, hidden_dim)
        return fused_features.mean(dim=1)  # (batch_size, hidden_dim)    


class SequenceDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_length=240 * 10, num_layers=4):
        super(SequenceDecoder, self).__init__()
        self.seq_length = seq_length  # n time steps
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8),
            num_layers=num_layers
        )
        self.position_embedding = nn.Embedding(seq_length, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, hidden_dim)
        batch_size = x.shape[0]
        
        positions = torch.arange(self.seq_length, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)  # (batch_size, seq_length, hidden_dim)
        
        x = x.unsqueeze(1).expand(-1, self.seq_length, -1)  # (batch_size, seq_length, hidden_dim)
        x = x + pos_emb  
        
        # Transformer expects (seq_length, batch_size, hidden_dim)
        x = x.permute(1, 0, 2)
        
        output = self.transformer_decoder(x, x)  # (seq_length, batch_size, hidden_dim)
        
        output = self.fc_out(output)  # (seq_length, batch_size, 4)
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, 4)
        output = output.view(output.shape[0], output.shape[1], 2, 2)  # (batch_size, seq_length, 2, 2)
        
        return output


# class AccidentSimulator(nn.Module):
#     def __init__(self, seq_length=10):
#         super(AccidentSimulator, self).__init__()
#         self.image_encoder = ImageEncoder()
#         self.text_encoder = TextEncoder()
#         self.fusion = MultiModalFusion(image_feat_dim=1792, text_feat_dim=768, hidden_dim=512)
#         self.decoder = SequenceDecoder(
#             input_dim=512, 
#             hidden_dim=512, 
#             output_dim=3,  # 3 controls per timestep
#             seq_length=seq_length
#         )

#     def forward(self, images, text):
#         image_features = self.image_encoder(images)
#         text_features = self.text_encoder(text)
#         fused_features = self.fusion(image_features, text_features)
#         controls = self.decoder(fused_features)  # (batch_size, seq_length, 3)
#         return controls

class AccidentSimulator(nn.Module):
    def __init__(self):
        super(AccidentSimulator, self).__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.fusion = MultiModalFusion(
            image_feat_dim=1792,  # Image feature dimension (output from EfficientNet B4)
            text_feat_dim=768,  # Text feature dimension (output from BERT)
            spawn_feat_dim=3,  # Spawn position feature dimension (x, y, z)
            hidden_dim=512  # Hidden dimension for projection
        )
        self.decoder = SequenceDecoder(input_dim=512, hidden_dim=512, output_dim=4)  # 4 outputs per timestep

    def forward(self, images, text, car1_spawn, car2_spawn):
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(text)
        spawn_positions = torch.cat([car1_spawn, car2_spawn], dim=1)  # (batch_size, 6)
        fused_features = self.fusion(image_features, text_features, spawn_positions)
        controls = self.decoder(fused_features)
        return controls
