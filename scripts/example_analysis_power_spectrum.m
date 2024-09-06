%% load an audio file 
% (here we also normalize the amplitude for presentation convenience)
filepath = '/Users/Yardenc/Documents/GitHub/Acoustic_analysis_lecture/data/CanarySong1.wav';
[y,samplerate] = audioread(filepath);
sig_in = y/(sqrt(mean(y.^2)));

%% display the audio
dt = 1/samplerate;
figure; plot(dt:dt:numel(y)*dt,y);
set(gca,'FontSize',16); xlabel('Time (sec)'); ylabel('Amplitude (au)'); title('Acoustic signal');

%% highpass filter the audio and display
y_highpass = highpass(y,500,samplerate,"StopbandAttenuation",40);
figure; plot(dt:dt:numel(y_highpass)*dt,y_highpass);
set(gca,'FontSize',16); xlabel('Time (sec)'); ylabel('Amplitude (au)'); title('Highpass-filtered acoustic signal');

%% Calculate the power spectrum
nfft = 2048;
noverlap = 0;
win = hamming(nfft); %w = w/sum(w)*numel(w);
% [Pxx,F] = pwelch(X,WINDOW,NOVERLAP,NFFT,Fs)
[Pxx,W] = pwelch(y_highpass,win,noverlap,nfft,samplerate);
figure; plot(W,Pxx);
set(gca,'FontSize',16); xlabel('Frequency (Hz)'); ylabel('PSD (au^2)'); title('Highpass-filtered acoustic signal');

%% Parseval's theorem (fft example).
N = 1000;
r = rand(1,N);
E1 = sum(r.^2)
g = fft(r);
E2 = sum(abs(g).^2)/N

%% Apply Parseval's theorem to our signal
disp(['Mean power = mean(signal^2) = ' num2str(mean(y_highpass.^2))]);
disp(['Sum of PSD = sum(PSD)*dF = ' num2str(sum(Pxx)*W(2))])

