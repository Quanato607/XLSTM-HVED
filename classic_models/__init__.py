from RA_HVED import U_HVEDNet3D, U_HVEDConvNet3D,  U_HVEDConvXLSTMNet3D, U_HVEDConvDuSFENet3D, U_HVEDConvDuSFESkrNet3D, U_HVEDConvDuSFEmViLNet3D, U_HVEDConvDuSFEmViLSkrNet3D, XLSTM_HVED_woViL,XLSTM_HVED_woDuSFE, XLSTM_HVED_woSMVAE

cfgs = {
    'U_HVEDNet3D': U_HVEDNet3D,
    'U_HVEDConvNet3D': U_HVEDConvNet3D,
    'U_HVEDConvXLSTMNet3D': U_HVEDConvXLSTMNet3D,
    'U_HVEDConvDuSFENet3D': U_HVEDConvDuSFENet3D,
    'U_HVEDConvDuSFESkrNet3D': U_HVEDConvDuSFESkrNet3D,
    'U_HVEDConvDuSFEmViLNet3D': U_HVEDConvDuSFEmViLNet3D,
    'U_HVEDConvDuSFEmViLNet3D_pretrain': U_HVEDConvDuSFEmViLNet3D,
    'U_HVEDConvDuSFEmViLSkrNet3D': U_HVEDConvDuSFEmViLSkrNet3D,
    'U_HVEDDuSFEmViLDFNet3D': U_HVEDConvDuSFEmViLSkrNet3D,
    'XLSTM_HVED': U_HVEDConvDuSFEmViLSkrNet3D,
    'XLSTM_HVED_woViL': XLSTM_HVED_woViL, 
    'XLSTM_HVED_woDuSFE': XLSTM_HVED_woDuSFE,
    'XLSTM_HVED_woSMVAE': XLSTM_HVED_woSMVAE
    
}

def find_model_using_name(model_name):   
    return cfgs[model_name]