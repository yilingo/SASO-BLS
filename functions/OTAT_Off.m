classdef OTAT_Off
    properties
        Name = 'other sa method';

    end
    %% Functions and algorithm
    methods    (Static = true)              
        function model = OT_SA(model,SelTrainA,num_resample,method)
            M = length(SelTrainA(1,:));
            xmin = min(SelTrainA);
            xmax = max(SelTrainA);
            SampStrategy = 'lhs' ;
            DistrFun  = 'unif'  ; 
            DistrPar=cell(M,1); for i=1:M; DistrPar{i}=[xmin(i) xmax(i)]; end
            if any(strcmpi(method,{'SV_SA','GV_SA'}))  
                tau = 0.00001;
                X = AAT_sampling(SampStrategy,M,DistrFun,DistrPar,2*num_resample);
                [ XA, XB, XC ] = vbsa_resampling(X) ;
                YA = XA * model.Beta ; % size (num_resample,10)
                YB = XB * model.Beta ; % size (num_resample,10)
                YC =XC * model.Beta ; % size (num_resample*M,10)
                for i = 1:length(YA(1,:))              
                    [ Si(:,i), STi(:,i)] = vbsa_indices(YA(:,i),YB(:,i),YC(:,i));
                end
                if strcmp(method,'SV_SA')
                    S_matrix = Si;
                elseif strcmp(method,'GV_SA')
                    S_matrix = STi;
                end
            elseif strcmp(method,'EET_SA')                 
                design_type='trajectory';   
                tau = 0.001;
                X = OAT_sampling(num_resample,M,DistrFun,DistrPar,SampStrategy,design_type);
                Y = X  * model.Beta;   
                for i =1:length(Y(1,:))
                    Yi = Y(:,i);
                    [ S_matrix(:,i), ~ ] = EET_indices(num_resample,xmin,xmax,X,Yi,design_type);            
                end
            end           
            FyFeature = max(S_matrix,[],2);           
            [FyFeature_sort, Sort_index] = sort(FyFeature,'descend');
            DeltaFeatureFy = zeros(length(FyFeature_sort)-1,1);
            for i = 1:(length(FyFeature_sort)-1)                
                DeltaFeatureFy(i) = FyFeature_sort(i)-FyFeature_sort(i+1);
            end
            MeanFeatureFy = mean(DeltaFeatureFy);
            Condition1 = DeltaFeatureFy./FyFeature_sort(1:end-1);
            % condition 1
            Condition1_judge = (Condition1>tau);
            SelectIndex1 = find(Condition1_judge==1);
            SelectNeuron1 = Sort_index(SelectIndex1);
            % condition 2
            Condition2_judge = (FyFeature_sort>1*MeanFeatureFy) ;
            SelectIndex2 = find(Condition2_judge==1);
            SelectNeuron2 = Sort_index(SelectIndex2);
            %  union
            SelectNeruonSet = intersect(SelectNeuron1,SelectNeuron2);          
            model.BanNodes = setdiff(Sort_index,SelectNeruonSet);                      
        end
    end %methodq
end %class
         
        

         
        
