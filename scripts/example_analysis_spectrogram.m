%%
time_step_ms = 1;
NW = 4; % Number of window function to create (even though we use 2)
nfft = 512;
[E,V] = dpss(nfft,NW); % we will use 2-taper Slepian windows
noverlap = nfft - round(time_step_ms/1000*samplerate);
[S1,F,T] = spectrogram(sig_in,E(:,1),noverlap,nfft,samplerate);
[S2,F,T] = spectrogram(sig_in,E(:,2),noverlap,nfft,samplerate);
S = S1.*conj(S1)+S2.*conj(S2); % the sonogram