classdef DataSetLoad
    properties
        Name = 'Load Dataset';
    end
    methods    (Static = true)      
        %% load data
        function [ForTrain_x,ForTest_x,ForTrain_y,ForTest_y] = load(DataName)
            if strcmp(DataName,'TE')
                load DataSet/TE_TrainData.mat;
                load DataSet/TE_TestData.mat;
                train_x = im2double(Train_data(:,1:52));
                test_x = im2double(Test_data(:,1:52));
                train_yc = im2double(Train_data(:,53));
                train_yc = MyClassTools.ChangeLabel(train_yc);
                test_yc = im2double(Test_data(:,53));
                test_yc = MyClassTools.ChangeLabel(test_yc);
                train_x(:,46) = [];
                test_x(:,46) = [];
            elseif strcmp(DataName,'Bao')
                load DataSet/BaoSteel_TrainData52_sel.mat;
                load DataSet/BaoSteel_ValData52_sel.mat;                
                train_x = BaoSteel_TrainData(:,1:end-1);
                test_x = BaoSteel_ValData(:,1:end-1);
                train_yc = BaoSteel_TrainData(:,end);  
                train_yc = MyClassTools.ChangeLabel(train_yc);
                test_yc = BaoSteel_ValData(:,end);
                test_yc = MyClassTools.ChangeLabel(test_yc);
            end
            ForTrain_x = train_x;
            ForTrain_y = train_yc;
            ForTest_x = test_x;
            ForTest_y = test_yc;            
        end

        %% shuffle data
        function [ForTrain_x,ForTrain_y] = Shuffle(ForTrain_x,ForTrain_y)
            rowrank = randperm(size(ForTrain_x,1));
            datasize = length(size(ForTrain_x));
            if  datasize == 2
                ForTrain_x = ForTrain_x(rowrank,:);
                ForTrain_y = ForTrain_y(rowrank,:);
            elseif datasize == 3
                ForTrain_x = ForTrain_x(rowrank,:,:);
                ForTrain_y = ForTrain_y(rowrank,:,:);
            end
        end

        %% normlize data
        function [ForTrain_x,ForTest_x] = Norm(ForTrain_x,ForTest_x,NormMethod)
            if NormMethod == 1
                [minn, maxx, ForTrain_x] = MyClassTools.NormDm(ForTrain_x);
                ForTest_x = MyClassTools.normDadaptm(ForTest_x, minn, maxx);
            elseif NormMethod == 2
                [minn, maxx, ForTrain_x] = MyClassTools.NormDp(ForTrain_x);
                ForTest_x = MyClassTools.normDadaptp(ForTest_x, minn, maxx);
            elseif NormMethod == 3
                ForTrain_x = zscore(ForTrain_x);
                ForTest_x = zscore(ForTest_x);
                [minn, maxx, ForTrain_x] = MyClassTools.NormDp(ForTrain_x);
                ForTest_x = MyClassTools.normDadaptp(ForTest_x, minn, maxx);
            elseif NormMethod == 4
                AllX = [ForTrain_x;ForTest_x];
                [~, ~, AllX] = MyClassTools.NormDp(AllX);
                ForTrain_x = AllX(1:size(ForTrain_x,1),:);
                ForTest_x = AllX(size(ForTrain_x,1)+1:end,:);
            elseif NormMethod == 5
                AllX = [ForTrain_x;ForTest_x];
                AllX = zscore(AllX);
                ForTrain_x = AllX(1:size(ForTrain_x,1),:);
                ForTest_x = AllX(size(ForTrain_x,1)+1:end,:);
            end
        end

        %% data segmentation for online experiment
        function [IniForTrain_x,IniForTrain_y,RestForTrain_x,RestForTrain_y] = OnlineSeg(ForTrain_x,ForTrain_y,OnlineIntrate,OnlineStep)
                NumEachLabelMat = tabulate(MyClassTools.ClassResult(ForTrain_y));  
                NumLabelClass = length(NumEachLabelMat(:,1));
                NumEachLabelList = NumEachLabelMat(:,2);
                NumEachLabelListInit = ceil(NumEachLabelList * OnlineIntrate); 
                NumEachLabelListLast = NumEachLabelList - NumEachLabelListInit; 
                NumEachLabelListStep = ceil(NumEachLabelListLast/OnlineStep);
                IniForTrain_x = [];
                IniForTrain_y = [];                         
                NumStart = 1;                
                RestForTrain_x = cell(1,OnlineStep);
                RestForTrain_y = cell(1,OnlineStep);
                for i = 1 : NumLabelClass                    
                    NumEnd = NumStart + NumEachLabelListInit(i) -1;
                    IniForTrain_x_i = ForTrain_x(NumStart:NumEnd,:);
                    IniForTrain_y_i = ForTrain_y(NumStart:NumEnd,:);
                    LastForTrain_x_i = ForTrain_x(NumEnd+1:sum(NumEachLabelList(1:i)),:);
                    LastForTrain_y_i = ForTrain_y(NumEnd+1:sum(NumEachLabelList(1:i)),:);
                    IniForTrain_x = [IniForTrain_x;IniForTrain_x_i];
                    IniForTrain_y = [IniForTrain_y;IniForTrain_y_i];  
                    NumStartStep = 1;
                    for j = 1: OnlineStep
                        if j ~=OnlineStep
                            NumEndStep = NumStartStep + NumEachLabelListStep(i) -1;
                        else
                            NumEndStep = NumEachLabelListLast(i);
                        end
                        LastForTrain_x_j = LastForTrain_x_i(NumStartStep:NumEndStep,:);
                        LastForTrain_y_j = LastForTrain_y_i(NumStartStep:NumEndStep,:);
                        RestForTrain_x{j} = [RestForTrain_x{j};LastForTrain_x_j];
                        RestForTrain_y{j} = [RestForTrain_y{j};LastForTrain_y_j];
                        NumStartStep = NumEndStep+1;
                    end
                    NumStart = sum(NumEachLabelList(1:i))+1;
                end                
        end

    end %method
end %class 
         
        

         
        
