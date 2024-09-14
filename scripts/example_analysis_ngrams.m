%% Lecture on analysis of acoustic behavior
% Yarden Cohen, 2024
% Djs, Ngram entropies

% Some code uses sparse matix variables found here: https://www.mathworks.com/matlabcentral/fileexchange/29832-n-dimensional-sparse-arrays 

%% required GitHub repositories:
Githubfolder = '/Users/Yardenc/Documents/GitHub/';
addpath(genpath(fullfile(Githubfolder,'BirdSongBout_YC'))); % clone: https://github.com/yardencsGitHub/BirdSongBout.git
addpath(genpath(fullfile(Githubfolder,'pst'))); % clone: https://github.com/yardencsGitHub/pst.git 

%% build transition probabilities
path_to_data = '/Users/Yardenc/Documents/GitHub/Acoustic_analysis_lecture/data/Annotated_songs_phrases.mat';
load(path_to_data);
ALPHABET = unique([DATA{:}]);
L=5;
try
    [f_mat alphabet n pi_dist]=pst_build_trans_mat_full(DATA,L,'alphabet',ALPHABET);
catch mem_em
    [f_mat alphabet n pi_dist]=pst_build_trans_mat(DATA,L,'alphabet',ALPHABET);
end
%% compare 1st order transition probabilites
% split the data in 2
DATA1 = DATA(1:floor(numel(DATA)/2)); 
DATA2 = DATA(floor(numel(DATA)/2)+1:end);
% calculate transition matrices
[f_mat1 alphabet n pi_dist]=pst_build_trans_mat_full(DATA1,1,'alphabet',ALPHABET);
[f_mat2 alphabet n pi_dist]=pst_build_trans_mat_full(DATA2,1,'alphabet',ALPHABET);
trans_mat_1 = double(f_mat1{2});
trans_mat_1 = trans_mat_1./(sum(trans_mat_1,2)*ones(1,numel(syllables))+1e-15);
trans_mat_2 = double(f_mat2{2});
trans_mat_2 = trans_mat_2./(sum(trans_mat_2,2)*ones(1,numel(syllables))+1e-15);
% remove the entries for sequence start and end
trans_mat_1 = trans_mat_1(3:end,2:end);
trans_mat_2 = trans_mat_2(3:end,2:end);

%% display transition probabilities from an example phrase type
phrase_n = 1;
P = trans_mat_1(phrase_n,:); Q = trans_mat_2(phrase_n,:);
figure; bar(1:30,P,'FaceAlpha',0.5); hold on;
bar(1.5:1:30.5,Q,'FaceAlpha',0.5);
set(gca,'FontSize',16); xlabel('Transition outcomes'); ylabel('Probability');
legend({'1^{st} data split', '2^{nd} data split'})

%% example Djs and entropies
M = 0.5*(P+Q);
d_kl1 = sum(P.*log2(P./(M+1e-20)+1e-20));
d_kl2 = sum(Q.*log2(Q./(M+1e-20)+1e-20));
Djs = 0.5*(d_kl1+d_kl2)
H1 = -sum(P.*log2(P+1e-20))
H2 = -sum(Q.*log2(Q+1e-20))

%%


Djs = small_utils_JS_divergence(trans_mat_1,trans_mat_2,'dim',2);
H1 = small_utils_ShannonEnt(trans_mat_1,'dim',2);
H2 = small_utils_ShannonEnt(trans_mat_2,'dim',2);
%% calculate Ngram entropies
ents = zeros(1,L);
for depth = 1:L
    x = f_mat{depth+1};
    x = double(x(x>0)); x = x/sum(x);
    ents(depth) = -sum(x.*log2(x+1e-20));
end


%% helper functions
function d = small_utils_JS_divergence(P,Q,varargin)
    dim = 1;
    nparams=length(varargin);
    if mod(nparams,2)>0
        error('Parameters must be specified as parameter/value pairs');
    end
    for i=1:2:nparams
        switch lower(varargin{i})
	        case 'dim'
		        dim=varargin{i+1};
        end
    end
    M = (P+Q)/2;
    d = (small_utils_KL_divergence(P,M,'dim',dim) + small_utils_KL_divergence(Q,M,'dim',dim))/2;
end

function d = small_utils_KL_divergence(P,Q,varargin)
    dim = 1;
    nparams=length(varargin);
    if mod(nparams,2)>0
        error('Parameters must be specified as parameter/value pairs');
    end
    for i=1:2:nparams
        switch lower(varargin{i})
	        case 'dim'
		        dim=varargin{i+1};
        end
    end
    d = sum(P.*log2(P./(Q+1e-20)+1e-20),dim);
end

function H = small_utils_ShannonEnt(p,varargin)
    dim = 1;
    nparams=length(varargin);
    if mod(nparams,2)>0
        error('Parameters must be specified as parameter/value pairs');
    end
    for i=1:2:nparams
        switch lower(varargin{i})
	        case 'dim'
		        dim=varargin{i+1};
        end
    end
    H = -sum(p.*log(p+1e-20),dim)/log(2);
end