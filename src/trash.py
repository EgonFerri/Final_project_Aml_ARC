class CNN(nn.Module):
    """
    CNN
    """
    
    def __init__(self, task_train, w, h, attention):
        super(CNN, self).__init__()

        self.inp_dim = np.array(np.array(task_train[0]['input']).shape)
        self.out_dim = np.array(np.array(task_train[0]['output']).shape)
        self.conv = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=5, padding=2)
        self.attention = Attention(10, 10)
        self.attention_value = attention

    def forward(self, x):  
        ch, input_w, input_h = x.shape[0], x.shape[1], x.shape[2]
        x = pad_crop(x, self.out_dim[0], self.out_dim[1], self.inp_dim[0], self.inp_dim[1], goal='pad')
        x = x.unsqueeze(0)
        #x = torch.nn.Upsample(size=(self.out_dim[0], self.out_dim[1]))(x)
        x = self.conv(x)
        if self.attention_value is not None:
          x = torch.reshape(x, (1, x.shape[2], x.shape[3], 10))
          x = self.attention(x)
          x = torch.reshape(x, (1, 10, x.shape[1], x.shape[2]))
        return x