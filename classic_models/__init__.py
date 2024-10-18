from RA_HVED import *
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
    'XLSTM_HVED': XLSTM_HVED,
    'XLSTM_HVED_woViL': XLSTM_HVED_woViL, 
    'XLSTM_HVED_woDuSFE': XLSTM_HVED_woDuSFE,
    'XLSTM_HVED_woSMVAE': XLSTM_HVED_woSMVAE,
    'XLSTM_HVED_drop_vil': XLSTM_HVED_drop_vil,
    'RA_HVED': RA_HVED,
    'XLSTM_HVED_missing1': XLSTM_HVED,
    'XLSTM_HVED_woME_VAEback': XLSTM_HVED_woME_VAEback,
    'XLSTM_HVED_woME_VAEback_woViL': XLSTM_HVED_woME_VAEback_woViL,
    'XLSTM_HVED_woME_VAEback_CK': XLSTM_HVED_woME_VAEback_CK,
    'XLSTM_HVED_woME_VAEback_ViLAtt': XLSTM_HVED_woME_VAEback_ViLAtt,
    'XLSTM_HVED_woME_VAEback_ViLAtt_woskip_vil_m1': XLSTM_HVED_woME_VAEback_ViLAtt,
    'XLSTM_HVED_woME_VAEback_ViLAtt_DC': XLSTM_HVED_woME_VAEback_ViLAtt,
    'XLSTM_HVED_woME_VAEback_ViLAtt_DC_noPretrain': XLSTM_HVED_woME_VAEback_ViLAtt

    
}

def find_model_using_name(model_name):   
    return cfgs[model_name]