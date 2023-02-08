%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% Comparation among BLS, BLS+FSA and BLS+TSA %%%%%%%%%%%%%%%%%
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
sigfun = 'logsig';
NormMethod = 4;
[train_x,test_x,ForTrain_y,ForTest_y] = DataSetLoad.load(DataSet);

%% data norm
[ForTrain_x,ForTest_x] = DataSetLoad.Norm(train_x,test_x,NormMethod);

%% Parameter set 
% model parameters
NumPerWin = 10;  % Nodes number of the feature mapping layer per window
NumWindow = 10;  % Nodes number of windows of the feature mapping layer
NumEnhPer = 100; % Nodes number of enhancement layer added per test

% incremental learning: None
NumFeaPerInc = [];
NumEnhRelPerInc = [];
NumEnhPerInc = [];

% other parameters
L2Param = 2^-30; %L2 parameter
ShrScale = .8;   %the l2 regularization parameter and the shrinkage scale of the enhancement nodes   
BanType = 'All'; %FeatureNodes %All %EnhanNodes
StartStep = 0;
BanIndex = [];
ThetaSel = 0.085;
InitMed = 'GuassX'; %MeanX , GuassX, MeanHe,GuassHe
ifplot = false;
test_step = 10;
Repet_time = 100;

%% Multi-test Process
idicators = {'BLS_Pre',   'BLS_Rec',   'BLS_F1',   'BLS_Par',   'BLS_time',...
             'BLSTSA_Pre','BLSTSA_Rec','BLSFSA_F1','BLSTSA_Par','BLSTSA_time',...
             'BLSFSA_Pre','BLSFSA_Rec','BLSTSA_F1','BLSFSA_Par','BLSFSA_time'};
for i = 1:length(idicators)
    eval([idicators{i} '=zeros(Repet_time,test_step)'])
end

for z = 1:Repet_time

    disp(['********Start the ', num2str(z), '-th round ********']);

    for i = 1:test_step
        % different number of Enhancement layer
        NumEnhance = i*NumEnhPer; 
        disp(['********Start the ', num2str(z), '-th round with ', num2str(NumEnhance), ' enhance nodes learning process********']);      
             

        % Model Initialization
        Model = SASO_Class(NumPerWin,NumWindow,NumEnhance,NumFeaPerInc,NumEnhRelPerInc,NumEnhPerInc,ShrScale,L2Param,BanIndex,StartStep,sigfun,InitMed,NormMethod);
    
        %% BLS training
        tic;
        Model = Model.Train(ForTrain_x,ForTrain_y); 
        BLS_time(z,i) = toc;
    
        % get BLS results
        TrainResult = Model.GetOutput(ForTrain_x);    
        TrainLabelDis = MyClassTools.ClassResult(ForTrain_y);
        ValResult = Model.GetOutput(ForTest_x);
        ValResultDis = MyClassTools.ClassResult(ValResult);
        ValLabelDis = MyClassTools.ClassResult(ForTest_y);
        ValIndex = Evaluation_idx(ValResultDis,ValLabelDis);
        [~,~,~,~,~,~,WMacro_P,WMacro_R,WMacro_F1] = ValIndex.Macro();
        BLSpara = ((length(ForTrain_x(1,:))+1)*NumPerWin*NumWindow+...
            (NumPerWin*NumWindow+1)*NumEnhance+(NumPerWin*NumWindow+NumEnhance+1)*10)/1000;
        
        % BLS output and save
        disp(['The Precision of BLS is ' ,num2str(WMacro_P)]);
        fprintf(2,['The Recall of BLS is ' ,num2str(WMacro_R),'\n']);
        fprintf(2,['The macro-F1 of BLS is ' ,num2str(WMacro_F1),'\n']);
        disp(['The parameter of BLS is ' ,num2str(BLSpara)]);
        BLS_Pre(z,i) = WMacro_P;
        BLS_Rec(z,i) = WMacro_R;
        BLS_F1(z,i) = WMacro_F1;
        BLS_Par(z,i) = BLSpara;
        
        %% SA preparation
        NumEachLabel = tabulate(TrainLabelDis);
        NumEech4SA = min(NumEachLabel(:,2));
        SelTrainA = Model.A_Matrix_Train;
        
        %% FPD-SA for compression
        tic;
        Model_FSA = FPD_SA_Off.SA(Model,SelTrainA,NumEech4SA,sigfun,ThetaSel); 
        BLSFSA_time(z,i) = toc;
    
        % get FPD-SA results
        [ModelFSA,~] = Model_FSA.PrunOutput(ForTrain_x,BanType,ForTrain_y,'update');  
        [~,FSAValResult] = ModelFSA.PrunOutput(ForTest_x,BanType,ForTest_y,'test');
        FSAValResultDis = MyClassTools.ClassResult(FSAValResult);
        FSAValIndex = Evaluation_idx(FSAValResultDis,ValLabelDis);
        [~,~,~,~,~,~,WMacro_P,WMacro_R,WMacro_F1]  = FSAValIndex.Macro();
    
        % get model param after FPD-SA 
        BanNumEnhanceFSA = length(Model_FSA.BanNodes(Model_FSA.BanNodes>NumPerWin*NumWindow));
        BanNumFeatureFSA = length(Model_FSA.BanNodes)-BanNumEnhanceFSA;
        NumEnhanceFSA = NumEnhance-BanNumEnhanceFSA;    
        NumFeatureFSA = NumPerWin*NumWindow-BanNumFeatureFSA;
        FSApara = ((length(ForTrain_x(1,:))+1)*NumFeatureFSA+...
            (NumFeatureFSA+1)*NumEnhanceFSA+(NumFeatureFSA+NumEnhanceFSA+1)*10)/1000;
        
        % FPD-SA output and save
        BLSFSA_Pre(z,i) = WMacro_P;
        BLSFSA_Rec(z,i) = WMacro_R;
        BLSFSA_F1(z,i) = WMacro_F1;
        BLSFSA_Par(z,i) = FSApara;
        disp(['The Precision of FSA is -------' ,num2str(WMacro_P)]);
        fprintf(2,['The Recall of FSA is -------' ,num2str(WMacro_R),'\n']);
        fprintf(2,['The macro-F1 of FSA is ' ,num2str(WMacro_F1),'\n']);
        disp(['The parameter of FSA is -------' ,num2str(FSApara)]);
    
        %% traditional partial differential SA for compression
        tic
        Model_TSA = PD_TSA_Off.TSA(Model,SelTrainA,sigfun); 
        BLSTSA_time(z,i) = toc;
    
        % get TSA results
        [ModelTSA,FSATrainResult] = Model_TSA.PrunOutput(ForTrain_x,BanType,ForTrain_y,'update');
        [~,FSAValResult] = ModelTSA.PrunOutput(ForTest_x,BanType,ForTest_y,'test');
        TSAValResultDis = MyClassTools.ClassResult(FSAValResult);
        TSAValIndex = Evaluation_idx(TSAValResultDis,ValLabelDis);
        [~,~,~,~,~,~,WMacro_P,WMacro_R,WMacro_F1]  = TSAValIndex.Macro();
    
        % get model param after TSA 
        BanNumEnhanceTSA = length(Model_TSA.BanNodes(Model_TSA.BanNodes>NumPerWin*NumWindow));
        BanNumFeatureTSA = length(Model_TSA.BanNodes)-BanNumEnhanceTSA;
        NumEnhanceTSA = NumEnhance - BanNumEnhanceTSA;
        NumFeatureTSA = NumPerWin*NumWindow-BanNumFeatureTSA;
        TSApara = ((length(ForTrain_x(1,:))+1)*NumFeatureTSA+...
            (NumFeatureTSA+1)*NumEnhanceTSA+(NumFeatureTSA+NumEnhanceTSA+1)*10)/1000;  
    
        % TSA output and save
        BLSTSA_Pre(z,i) = WMacro_P;
        BLSTSA_Rec(z,i) = WMacro_R;
        BLSTSA_F1(z,i) = WMacro_F1;
        BLSTSA_Par(z,i) = FSApara;
        disp(['The Precision of TSA is --------------' ,num2str(WMacro_P)]);
        fprintf(2,['The Recall of TSA is --------------' ,num2str(WMacro_R),'\n']);
        fprintf(2,['The macro-F1 of TSA is --------------' ,num2str(WMacro_F1),'\n']);
        disp(['The parameter of TSA is --------------' ,num2str(TSApara)]);
    end
end

file = 'Results\TE\SACF\';
mkdir (file);
file_name = ['SACF_',num2str(Repet_time),'.mat'];
save([file,file_name], 'BLS_Pre',   'BLS_Rec',   'BLS_F1',   'BLS_Par',   'BLS_time',...
           'BLSTSA_Pre','BLSTSA_Rec','BLSFSA_F1','BLSTSA_Par','BLSTSA_time',...
           'BLSFSA_Pre','BLSFSA_Rec','BLSTSA_F1','BLSFSA_Par','BLSFSA_time')

%% plot recall, precision, maroc-F1 , time, parameter 
if ifplot
    nodes_line = 1:1:test_step;
    BLS_mean = [mean(BLS_Rec);mean(BLS_Pre);mean(BLS_F1);mean(BLS_time);mean(BLS_Par)];
    BLSFSA_mean = [mean(BLSFSA_Rec);mean(BLSFSA_Pre);mean(BLSFSA_F1);mean(BLSFSA_time);mean(BLSFSA_Par)];
    BLSTSA_mean = [mean(BLSTSA_Rec);mean(BLSTSA_Pre);mean(BLSTSA_F1);mean(BLSTSA_time);mean(BLSTSA_Par)];       

    for i =1:5
        subplot(5,1,i);
        % BLS        
        plot(BLS_mean(i,:),'b')
        hold on;
    
        % BLS + FPD_SA 
        plot(BLSFSA_mean(i,:),'r')

        
        % BLS + traditional SA
        plot(BLSTSA_mean(i,:),'g')

    end

end

disp('Finish the Demo!')
