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
ThetaSel = 0.07;
L2Param = 2^-30; %L2 parameter
ShrScale = .8;   %the l2 regularization parameter and the shrinkage scale of the enhancement nodes   
BanType = 'All'; %FeatureNodes %All %EnhanNodes
StartStep = -1;
Step = 50;
BanIndex = [];
InitMed = 'GuassX'; %MeanX , GuassX, MeanHe,GuassHe
ifplot = false;

%% Model Initialization
Model = SASO_Class(NumPerWin,NumWindow,NumEnhance,NumFeaPerInc,NumEnhRelPerInc,NumEnhPerInc,ShrScale,L2Param,BanIndex,StartStep,sigfun,InitMed);

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
    SASOBLS_para = MyClassTools.bls_parameters(Model,'bls','offline');

    BLS_Pre(1,Model.Step+1) = WMacro_P;
    BLS_Rec(1,Model.Step+1) = WMacro_R;
    BLS_F1(1,Model.Step+1) = WMacro_F1;
    BLS_Par(1,Model.Step+1) = SASOBLS_para;
    disp(['The Precision of BLS is ' ,num2str(WMacro_P)]);
    fprintf(2,['The Recall of BLS is ' ,num2str(WMacro_R),'\n']);
    fprintf(2,['The macro-F1 of BLS is ' ,num2str(WMacro_F1),'\n']);
    disp(['The parameter of BLS is ' ,num2str(SASOBLS_para),'K']);

    NumEachLabel = tabulate(MyClassTools.ClassResult(ForTrain_y));
    NumEech4SA = min(NumEachLabel(:,2));
    SelTrainA = Model.A_Matrix_Train;

    % SASO-BLS
    tic;
    Model = FPD_SA_Off.SA(Model,SelTrainA,NumEech4SA,sigfun,ThetaSel); 
    SASOBLS_time(1,Model.Step+1) = toc;


    [Model_SASOBLS,~] = Model.PrunOutput(ForTrain_x,BanType,ForTrain_y,'update');
    [~,SASOBLS_ValResult] = Model_SASOBLS.PrunOutput(ForTest_x,BanType,ForTest_y,'test');
    FSAValResultDis = MyClassTools.ClassResult(SASOBLS_ValResult);
    FSAValIndex = Evaluation_idx(FSAValResultDis,ValLabelDis);
    [~,~,~,~,~,~,WMacro_P,WMacro_R,WMacro_F1]  = FSAValIndex.Macro();
    SASOBLS_para = MyClassTools.bls_parameters(Model,'saso-bls','offline');

    % FPD-SA output and save
    SASOBLS_Pre(1,Model.Step+1) = WMacro_P;
    SASOBLS_Rec(1,Model.Step+1) = WMacro_R;
    SASOBLS_F1(1,Model.Step+1) = WMacro_F1;
    SASOBLS_Par(1,Model.Step+1) = SASOBLS_para;
    disp(['The Precision of SASO-BLS is -------' ,num2str(WMacro_P)]);
    fprintf(2,['The Recall of SASO-BLS is -------' ,num2str(WMacro_R),'\n']);
    fprintf(2,['The macro-F1 of SASO-BLS is ' ,num2str(WMacro_F1),'\n']);
    disp(['The parameter of SASO-BLS is -------' ,num2str(SASOBLS_para),'K']);

    if Model.Step<Step && WMacro_R > 0.75
        Model_SASOBLS.A_Matrix_Train = [];
        Model_SASOBLS.A_Inverse = [];
        Model_SASOBLS.TotFeaSpa = [];
        Model_SASOBLS.FeaPD = [];
        Model_SASOBLS.AllPD = [];
        save save_model\SASOBLS_offline Model_SASOBLS
        break;
    elseif Model.Step == Step
        Model = SASO_Class(NumPerWin,NumWindow,NumEnhance,NumFeaPerInc,NumEnhRelPerInc,NumEnhPerInc,ShrScale,L2Param,BanIndex,StartStep,sigfun,InitMed);
    end
end

%% plot result
if ifplot
    BLS_results = [BLS_Rec;BLS_Pre;BLS_F1;BLS_time;BLS_Par];
    SASOBLS_results = [SASOBLS_Rec;SASOBLS_Pre;SASOBLS_F1;SASOBLS_time;SASOBLS_Par];
    for i =1:5
        subplot(5,1,i);
        % BLS        
        plot(BLS_results(i,:),'b')
        hold on;

        % SASO-BLS 
        plot(SASOBLS_results(i,:),'r')
    end

end

% plot confusion matrix of saso-bls
% C=repmat(max(SASOBLS_ValResult')',1,length(SASOBLS_ValResult(1,:)));
% SASOBLS_ValResult(SASOBLS_ValResult<C)=0;
% SASOBLS_ValResult(SASOBLS_ValResult~=0)=1;
% plotconfusion(ForTest_y',SASOBLS_ValResult');
   
disp('Finish the Demo!')