
class Decoder(nn.Module):
    def __init__(
        self,
        d_emb_dim: int,
        xformer_heads: int,
        cardinality: List,
        count_numerical_attr:int,
        device,
        final_dropout:float = 0.2,
        num_xformer_layers:int = 1, 
        
    ):
        super().__init__()
        self.device = device
        self.d_emb_dim = d_emb_dim
        
        _dec_layer = nn.TransformerEncoderLayer(
            d_model = self.d_emb_dim , 
            nhead = xformer_heads
        )
        
        self.decoder_xformer_joint = nn.TransformerEncoder(
            _dec_layer, 
            num_layers = num_xformer_layers
        )
        
        # Projection layer  for each category
        self.num_cat = len(cardinality)
      

        self.op_decision_layer_cnn = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=1, kernel_size=[2,1]) for i in range(self.num_cat+1)]
        )
        
        self.op_decision_layer_linear = nn.ModuleList([
            nn.Sequential(  nn.Dropout(final_dropout),
                            nn.Linear(d_emb_dim,2),
                            nn.ReLU(),
                            nn.Softmax(dim=-1)
                         )
            for i in range(self.num_cat+1)])
        
        return
    
    def forward(self, encoder_input, emb_input):
        print(encoder_input.shape)
        encoder_input = self.decoder_xformer_joint(encoder_input)
        
        bs = encoder_input.size()[0]
        seq_len = encoder_input.shape[1]
        dim = encoder_input.size()[-1]
        
        # Stack the outputs vertically : shape [Batch, seq_len, 2, dim]
        x2 = torch.stack([encoder_input, emb_input],dim=-2)
        
        # N,C,H,W
        x3 = [self.op_decision_layer_cnn[i](x2[:,i,:,:].reshape(bs,1,2,dim)) for i in range(seq_len)] 
        x4 = [self.op_decision_layer_linear[i](x3[i]) for i in range(seq_len)]
        # Perform argmax to get labels
        x5 = [ torch.argmax(x4[i], dim=-1) for i in range(seq_len)]
        op = torch.stack(x5 ,dim=-1).squeeze()
        return op
    