from torch import optim

from Modules.TCDF_module import TCDFModule
from Networks.TCDF_net_validation import ADDSTCN_validation


class TCDFModule_validation(TCDFModule):
    def __init__(self, in_channels, levels, kernel_size, dilation, device, lr, epochs, confidence_s=0.8):
        super().__init__(in_channels,levels,kernel_size,dilation,device,lr,epochs,confidence_s)
        self.network = ADDSTCN_validation(in_channels=in_channels, levels=self.levels, kernel_size=kernel_size,
                               dilation=self.dilation, device=device).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)