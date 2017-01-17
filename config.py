'''
SUMMARY:  config file
AUTHOR:   Qiuqiang Kong
Created:  2016.10.15
Modified: 2017.01.17 delete unnecessary code
--------------------------------------
'''

# warblrb dataset
wbl_root_fd = "/vol/vssp/msos/qk/warblrb10k_public_wav"
wbl_wav_fd = wbl_root_fd + "/wav"
wbl_csv_path = wbl_root_fd + "/warblrb10k_public_metadata.csv"

# test dataset
test_wav_fd = "/vol/vssp/msos/qk/test_bird_wav"

# your workspace
scrap_fd = "/vol/vssp/msos/qk/bird_detection_scrap"     # you need modify this path

# wbl dataset workspace
wbl_cv10_csv_path = scrap_fd + "/warblrb_cv10.csv"
wbl_fe_fd = scrap_fd + "/wbl_fe"
wbl_denoise_wav_fd = scrap_fd + '/wbl_denoise_wav'
wbl_denoise_fe_fd = scrap_fd + '/wbl_denoise_fe'
wbl_denoise_fe_fft_fd = wbl_denoise_fe_fd + '/wbl_denoise_fe_fft'
wbl_denoise_fe_enhance_fft_fd = wbl_denoise_fe_fd + "/wbl_denoise_fe_enhance_fft_fd"
wbl_denoise_fe_enhance_mel_fd = wbl_denoise_fe_fd + "/wbl_denoise_fe_enhance_mel"
wbl_denoise_fe_enhance_pool_fft_fd = wbl_denoise_fe_fd + "/wbl_denoise_fe_enhance_pool_fft"
wbl_dev_md_fd = scrap_fd + "/wbl_dev_md"

# test dataset workspace
test_wav_fd = "/vol/vssp/msos/qk/test_bird_wav"
test_denoise_wav_fd = scrap_fd + "/test_denoise_wav"
test_denoise_fe_fd = scrap_fd + "/test_denoise_fe"
test_denoise_fe_fft_fd = test_denoise_fe_fd + "/test_denoise_fe_fft"
test_denoise_fe_enhance_fft_fd = test_denoise_fe_fd + "/test_denoise_fe_enhance_fft_fd"
test_denoise_fe_enhance_mel_fd = test_denoise_fe_fd + "/test_denoise_fe_enhance_mel"
test_denoise_fe_enhance_pool_fft_fd = test_denoise_fe_fd + "/test_denoise_fe_enhance_pool_fft"


# global params
win = 1024
fs = 44100.
n_duration = 440    # 44 frames per second, all together 10 seconds