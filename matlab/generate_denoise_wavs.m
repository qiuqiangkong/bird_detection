% SUMMARY:  denoise warblrb & ff1010 bird song wav
% AUTHOR:   QIUQIANG KONG
% Created:  2016.11.01
% Modified: - 
% -----------------------------------------------------------
function generate_denoise_wavs
clc
clear all
close all

addpath( '/user/HS229/qk00006/my_code2015.5-/matlab/toolbox/voicebox' )

% % generate denoise wavs for warblrb
generate_denoise_wbl_wavs();

% % generate denoise wavs for ff1010
% generate_denoise_ff1010_wavs();

generate_denoise_test_wavs();

end


% generate denoise wavs for warblrb
function generate_denoise_wbl_wavs()
    wbl_root_fd = '/vol/vssp/msos/qk/warblrb10k_public_wav';
    wbl_wav_fd = strcat( wbl_root_fd, '/wav' );
    wbl_csv_file = strcat( wbl_root_fd, '/warblrb10k_public_metadata.csv' );
    wbl_denoise_wav_fd = '/vol/vssp/msos/qk/bird_detection_scrap/wbl_denoise_wav';
    mkdir( wbl_denoise_wav_fd );
    gen_denoise_wavs( wbl_wav_fd, wbl_denoise_wav_fd );
end


% generate denoise wavs for ff1010
function generate_denoise_ff1010_wavs()
    ff1010_root_fd = '/vol/vssp/msos/qk/ff1010_bird'
    ff1010_wav_fd = strcat( ff1010_root_fd, '/wav' )
    ff1010_csv_file = strcat( ff1010_root_fd, '/ff1010bird_metadata.csv' )
    ff1010_denoise_wav_fd = '/vol/vssp/msos/qk/bird_detection_scrap/ff1010_denoise_wav'
    mkdir( ff1010_denoise_wav_fd );
    gen_denoise_wavs( ff1010_wav_fd, ff1010_denoise_wav_fd );
end

function generate_denoise_test_wavs()
    % YOU NEED MODIFY THE PATHS BELOW!
    test_wav_fd = '/vol/vssp/msos/qk/test_bird_wav';
    test_denoise_wav_fd = '/vol/vssp/msos/qk/bird_detection_scrap/test_denoise_wav';
    mkdir( test_denoise_wav_fd );
    gen_denoise_wavs( test_wav_fd, test_denoise_wav_fd );
end


function gen_denoise_wavs( wav_fd, out_wav_fd )
    names = dir( strcat( wav_fd, '/*.wav' ) );
    for n = 1:length(names)
        n
        out_path = strcat( out_wav_fd, '/', names(n).name );
        if ~exist(out_path, 'file')
            path = strcat( wav_fd, '/', names(n).name )
            [s, fs] = readwav( path );
  
            s = fliplr(s')';
            audiowrite('flirlp.wav', s, fs);

            [in, out] = run_omlsa('flirlp', 'zzz');
            [s, fs] = readwav('zzz.wav');
            s = fliplr(s')';
           
            audiowrite( out_path, s, fs );
        end
    end

end