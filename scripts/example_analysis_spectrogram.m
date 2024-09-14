%% Lecture on analysis of acoustic behavior
% Yarden Cohen, 2024
% Spectrogram, amplitude envelope, window functions

%% load an audio file 
% (here we also normalize the amplitude for presentation convenience)
filepath = '/Users/Yardenc/Documents/GitHub/Acoustic_analysis_lecture/data/CanarySong1.wav';
[y,samplerate] = audioread(filepath);
sig_in = y/(sqrt(mean(y.^2)));

%% highpass filter the audio and display
sig_in = highpass(sig_in,500,samplerate,"StopbandAttenuation",40);

%% Create a simple spectrogram 
time_step_ms = 1;
nfft = 512;
noverlap = nfft - round(time_step_ms/1000*samplerate);
win = nfft; %hamming(nfft);

[S,F,T] = spectrogram(sig_in,win,noverlap,nfft,samplerate);
figure; imagesc(T,F,abs(S)); axis xy; colormap(1-gray);
ylim([0 8000]); caxis([0,50]);
set(gca,'FontSize',16); xlabel('Time (sec)'); ylabel('Frequency (Hz)'); title('Spectrogram');
xlim([12.0329   14.6405]);

%% Create spectrogram using the Slepian window functions
% aka Prolate spheroidal wave functions
% maximizes the energy in the main lobe

time_step_ms = 1;
NW = 4; % Number of window function to create (even though we use 2)
nfft = 512;
[E,V] = dpss(nfft,NW); % we will use 2-taper Slepian windows
noverlap = nfft - round(time_step_ms/1000*samplerate);
[S1,F,T] = spectrogram(sig_in,E(:,1),noverlap,nfft,samplerate);
[S2,F,T] = spectrogram(sig_in,E(:,2),noverlap,nfft,samplerate);
S = S1.*conj(S1)+S2.*conj(S2); % the sonogram

figure; imagesc(T,F,abs(S)); axis xy; colormap(1-gray);
ylim([0 8000]); caxis([0,50]);
set(gca,'FontSize',16); xlabel('Time (sec)'); ylabel('Frequency (Hz)'); title('Spectrogram');
xlim([12.0329   14.6405]);
set(gcf,'Position',[555   448   963   175])

%% Calculate the power amplitude from the spectrogram
Ampl = sum(abs(S));
figure; plot(T,Ampl);
set(gca,'FontSize',16); xlabel('Time (sec)'); ylabel('Amp (au^2)');
xlim([12.0329   14.6405]);
set(gcf,'Position',[555   448   963   175]);
%% Plot a variety of window shapes.

NW = 4; % Number of window function to create (even though we use 2)
nfft = 512;
[E,V] = dpss(nfft,NW);
slep1 = E(:,1)/sum(E(:,1));
gwin1 = gausswin(nfft)/sum(gausswin(nfft));
hmwin1 = hamming(nfft)/sum(hamming(nfft));
hnwin1 = hann(nfft)/sum(hann(nfft));
figure; hold on;
plot(slep1);
plot(gwin1);
plot(hmwin1);
plot(hnwin1);
set(gca,'FontSize',16); xlabel('Time bin'); ylabel('Amplitude (au)'); title('Window shapes');
legend({'Slepian','Gaussian','Hamming','Hanning'});
