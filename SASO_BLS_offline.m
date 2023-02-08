%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%% SASO-BLS offline mode  %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2022

clear;
clc;
close all; 
warning off all;
format compact;

%% add path
addpath('utils');
addpath('functions');

%% DataSet Load 
DataSet = 'TE'; 
[Train_x,Test_x,ForTrain_y,ForTest_y] = DataSetLoad.load(DataSet);

%% data norm
NormMethod = 4;
[ForTrain_x,ForTest_x] = DataSetLoad.Norm(Train_x,Test_x,NormMethod);


%% Parameter set 
NumEnhance = 10; % Nodes number of the enhancement layer 
NumPerWin = 10;  %Nodes number of the feature mapping layer per window
NumWindow = 1;  % Number of windows of the feature mapping layer

% incremental learning
NumFeaPerInc = 10;
NumEnhRelPerInc = 10;
NumEnhPerInc = 10;

% other parameters
sigfun = 'tansig';
ThetaSel = 0.1;
L2Param = 2^-30; %L2 parameter
ShrScale = .8;   %the l2 regularization parameter and the shrinkage scale of the enhancement nodes   
BanType = 'All'; %FeatureNodes %All %EnhanNodes
StartStep = -1;
Step = 30;
BanIndex = [];
InitMed = 'GuassX'; %MeanX , GuassX, MeanHe,GuassHe
ifplot = false;

%% Model Initialization
Model = SASO_Class(NumPerWin,NumWindow,NumEnhance,NumFeaPerInc,NumEnhRelPerInc,NumEnhPerInc,ShrScale,L2Param,BanIndex,StartStep,sigfun,InitMed,NormMethod);
% Result sequence initalization

idicators = {'BLS_Pre',   'BLS_Rec',   'BLS_F1',   'BLS_Par',   'BLS_time',...
             'BLSTSA_Pre','BLSTSA_Rec','BLSFSA_F1','BLSTSA_Par','BLSTSA_time',...
             'BLSFSA_Pre','BLSFSA_Rec','BLSTSA_F1','BLSFSA_Par','BLSFSA_time'};
for i = 1:length(idicators)
    eval([idicators{i} '=zeros(1,Step+1);']);
end

while Model.Step <= Step
    Model.Step = Model.Step +1;    
    disp(['********Start the ', num2str(Model.Step), '-th learning process********']);

    %% BLS training
    if Model.Step == 0
        tic;
        Model = Model.Train(ForTrain_x,ForTrain_y); 
        BLS_time(1,Model.Step+1) = toc;
    else
        tic;
        Model = Model.IncBLS(ForTrain_x,ForTrain_y);
        BLS_time(1,Model.Step+1) = toc;
    end

    ValResult = Model.GetOutput(ForTest_x);
    ValResultDis = MyClassTools.ClassResult(ValResult);
    ValLabelDis = MyClassTools.ClassResult(ForTest_y);
    ValIndex = Evaluation_idx(ValResultDis,ValLabelDis);
    [~,~,~,~,~,~,WMacro_P,WMacro_R,WMacro_F1] = ValIndex.Macro();
    BLSpara = MyClassTools.bls_parameters(Model,'bls','offline');

    BLS_Pre(1,Model.Step+1) = WMacro_P;
    BLS_Rec(1,Model.Step+1) = WMacro_R;
    BLS_F1(1,Model.Step+1) = WMacro_F1;
    BLS_Par(1,Model.Step+1) = BLSpara;
    disp(['The Precision of BLS is ' ,num2str(WMacro_P)]);
    fprintf(2,['The Recall of BLS is ' ,num2str(WMacro_R),'\n']);
    fprintf(2,['The macro-F1 of BLS is ' ,num2str(WMacro_F1),'\n']);
    disp(['The parameter of BLS is ' ,num2str(BLSpara),'K']);

    NumEachLabel = tabulate(MyClassTools.ClassResult(ForTrain_y));
    NumEech4SA = min(NumEachLabel(:,2));
    SelTrainA = Model.A_Matrix_Train;

    % FPD-SA
    tic;
    Model = FPD_SA_Off.SA(Model,SelTrainA,NumEech4SA,sigfun,ThetaSel); 
    BLSFSA_time(1,Model.Step+1) = toc;


    [ModelFSA,~] = Model.PrunOutput(ForTrain_x,BanType,ForTrain_y,'update');
    [~,FSAValResult] = ModelFSA.PrunOutput(ForTest_x,BanType,ForTest_y,'test');
    FSAValResultDis = MyClassTools.ClassResult(FSAValResult);
    FSAValIndex = Evaluation_idx(FSAValResultDis,ValLabelDis);
    [~,~,~,~,~,~,WMacro_P,WMacro_R,WMacro_F1]  = FSAValIndex.Macro();
    FSABLSpara = MyClassTools.bls_parameters(Model,'saso-bls','offline');

    % FPD-SA output and save
    BLSFSA_Pre(1,Model.Step+1) = WMacro_P;
    BLSFSA_Rec(1,Model.Step+1) = WMacro_R;
    BLSFSA_F1(1,Model.Step+1) = WMacro_F1;
    BLSFSA_Par(1,Model.Step+1) = FSABLSpara;
    disp(['The Precision of FSA is -------' ,num2str(WMacro_P)]);
    fprintf(2,['The Recall of FSA is -------' ,num2str(WMacro_R),'\n']);
    fprintf(2,['The macro-F1 of FSA is ' ,num2str(WMacro_F1),'\n']);
    disp(['The parameter of FSA is -------' ,num2str(FSABLSpara),'K']);


end

%% plot result
if ifplot
    BLS_ = [BLS_Rec;BLS_Pre;BLS_F1;BLS_time;BLS_Par];
    BLSFSA_ = [BLSFSA_Rec;BLSFSA_Pre;BLSFSA_F1;BLSFSA_time;BLSFSA_Par];
    for i =1:5
        subplot(5,1,i);
        % BLS        
        plot(BLS_(i,:),'b')
        hold on;

        % SASO-BLS 
        plot(BLSFSA_(i,:),'r')
    end

end

% plot confusion matrix of saso-bls
C=repmat(max(FSAValResult')',1,length(FSAValResult(1,:)));
FSAValResult(FSAValResult<C)=0;
FSAValResult(FSAValResult~=0)=1;
plotconfusion(ForTest_y',FSAValResult');
   
disp('Finish the Demo!')



