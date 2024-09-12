%% Lecture on analysis of acoustic behavior
% Yarden Cohen, 2024
% Transition diagram, PSTs

% Some code uses sparse matix variables found here: https://www.mathworks.com/matlabcentral/fileexchange/29832-n-dimensional-sparse-arrays 

%% required GitHub repositories:
Githubfolder = '/Users/Yardenc/Documents/GitHub/';
addpath(genpath(fullfile(Githubfolder,'BirdSongBout_YC'))); % clone: https://github.com/yardencsGitHub/BirdSongBout.git
addpath(genpath(fullfile(Githubfolder,'pst'))); % clone: https://github.com/yardencsGitHub/pst.git 
%% Load an annotation file and create a transition matrix
path_to_data = '/Users/Yardenc/Documents/GitHub/Acoustic_analysis_lecture/data/Annotated_songs_syllables.mat';
load(path_to_data); 
num_syls = numel(syllables);
transP = zeros(num_syls);
AlphaNumeric = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';
AlphaNumeric = ['1' AlphaNumeric(1:num_syls-2) '2'];
alld = [DATA{:}];
for s1 = 1:numel(AlphaNumeric)-1
    for s2 = 2:numel(AlphaNumeric)
        transP(s1,s2) = sum(alld(find(alld == AlphaNumeric(s1)) + 1) == AlphaNumeric(s2))/sum(alld == AlphaNumeric(s1));
    end
end

%% Display the transition matrix as a graph
text_scale = 1.1;
text_offset = [-0.025,0];
figure('Position',[1968        1         859         836]);
P1 = transP;
for i = 1:num_syls-1
    nrm = sum(P1(i,:))-P1(i,i) + 1e-10;
    P1(i,:) = P1(i,:)/nrm;
    P1(i,i) = 0.00;
end
P1(P1<0.01) = 0; % Don't display very small transition probbilities
G=digraph(P1);
lbls = split(AlphaNumeric,''); lbls = lbls(2:end-1);
LWidths = 5*G.Edges.Weight/max(G.Edges.Weight);
h=plot(G,'NodeLabel',{},'Layout','circle','NodeFontSize',24,'LineWidth',LWidths,'Marker','o',...
    'MarkerSize',24,'NodeFontAngle','normal','ArrowPosition',0.9,'ArrowSize',10); 
text(h.XData*text_scale+text_offset(1),h.YData*text_scale+text_offset(2),lbls,'FontSize',16);

%% Probabilistic Suffix Trees 
% Find the optimal p_min for the data we have
% We run a 10-fold cross validation.
% We fit the model using 90% of the data and calculate the train and test set log likelihoods. 
% In this example we use 1000 songs.
path_to_data = '/Users/Yardenc/Documents/GitHub/Acoustic_analysis_lecture/data/Annotated_songs_phrases.mat';
load(path_to_data);
L = 6; % the longest context we allow.
[Liks,ps] = cross_validate_PST(DATA(1001:2000),L,'use_full',1);
% Show the cross validated parameter sensitivity
figure; semilogx(ps,-nanmean(squeeze(Liks.test_logl),2));
set(gca,'FontSize',16); xlabel('p_{min}'); ylabel('Test set likelihood (-log)'); title('Cross validating parameters');

%% build a PST
pm = 0.005; % we use the cross validated parameter.
L = 6;
ab = unique([DATA{:}]);
[f_mat alphabet n pi_dist]=pst_build_trans_mat_full(DATA,L,'alphabet',ab);
tree=pst_learn(f_mat,alphabet,n,'g_min',.01,'p_min',pm,'r',1.6,'alpha',17.5,'L',L);

%% display the PST
% This will create all pie charts and place them in one figure.
% The figure will need some graphics work to make pretty.
pie_colors_file_path = '/Users/Yardenc/Documents/GitHub/Acoustic_analysis_lecture/scripts/pie_colors.mat';
fh = bsb_create_complete_PST_fig(tree,orig_syls,ab,'pie_colors_file_path',pie_colors_file_path,'slice_fsize',0);


%% helper functions

function [Liks,ps] = cross_validate_PST(DATA_in,L,varargin)
    p_min = [0.0001 0.00015 0.001:0.001:0.01 0.02 0.05];
    nparams=length(varargin);
    if mod(nparams,2)>0
        error('Parameters must be specified as parameter/value pairs');
    end
    for i=1:2:nparams
        switch lower(varargin{i})
            case 'p_min'
		        p_min=varargin{i+1};
        end
    end
    
    ps = p_min;
    try
        Liks=pst_cross_validate(DATA_in,'p_min',p_min,'L',L,'use_full',1);
    catch mem_em
        Liks=pst_cross_validate(DATA_in,'p_min',p_min,'L',L);
    end
    
    
end
