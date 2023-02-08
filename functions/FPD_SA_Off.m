%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fast Partial Differential-based Sensitivity Analysis (Matlab)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2022
classdef FPD_SA_Off
    properties
        Name = 'Off-line Broad Learning System';
    end

    %% Functions and algorithm
    methods    (Static = true)              
        function model = SA(model,SelTrainA,NumEech4SA,sigfun,ThetaSel)
        %% parameter initial
            NumFeaOri = model.NumPerWin * model.NumWindow;
            NumEnhOri = model.NumEnhance; 
            EnhMatSelOri = SelTrainA(:,NumFeaOri + 1:NumFeaOri+NumEnhOri);
            NumClass = length(model.Beta(1,:));
            if model.Step == 0      
                AddFeaPDD = model.Beta(1:NumFeaOri,:);
                AddEnhPDD = model.Beta(NumFeaOri + 1:NumFeaOri+NumEnhOri,:); 
                AddEnhSel = EnhMatSelOri;
                AddRelSel = [];
                AddFeaEnhWei = model.FeaEnhWei(1:NumFeaOri,:);
                OriFeaAddEnhWei = [];
                AddFeaRelWei = [];                
                AddRelPDD = [];              
            else
                NumAddFea = model.NumAddFea;
                NumAddRel = model.NumAddRel;
                NumAddEnh = model.NumAddEnh;
                AddFeaPDD = model.Beta(end - (NumAddFea+NumAddEnh+NumAddRel) + 1:end - (NumAddEnh+NumAddRel),:);
                AddEnhSel = SelTrainA(:,end-NumAddEnh+1:end); 
                AddRelSel = SelTrainA(:,end-NumAddRel-NumAddEnh+1:end-NumAddEnh);  
                AddFeaEnhWei = model.AllFeaAddEnhWei{model.Step}(end-NumAddFea:end-1,:);
                OriFeaAddEnhWei = model.AllFeaAddEnhWei{model.Step}(1:end-NumAddFea-1,:);
                AddFeaRelWei = model.AddFeaRelWei{model.Step}(1:NumAddFea,:);                
                AddRelPDD = model.Beta(end - NumAddRel - NumAddEnh+1:end -NumAddEnh,:);
                AddEnhPDD = model.Beta(end-NumAddEnh+1:end,:); 
            end 

            %% select important nodes
            SelectNeruonSet = [];
            for z = 1:NumClass  
                % calculate partial differential
                if strcmp(sigfun,'logsig')
                    DiagAddEnhAll = AddEnhSel((z-1)*NumEech4SA+1:z*NumEech4SA,:) .*(1-AddEnhSel((z-1)*NumEech4SA+1:z*NumEech4SA,:));                    
                    try 
                        DiagAddRelAll = AddRelSel((z-1)*NumEech4SA+1:z*NumEech4SA,:) .*(1-AddRelSel((z-1)*NumEech4SA+1:z*NumEech4SA,:));
                    catch
                        DiagAddRelAll = [];
                    end
                else
                    DiagAddEnhAll = 1-AddEnhSel((z-1)*NumEech4SA+1:z*NumEech4SA,:) .* AddEnhSel((z-1)*NumEech4SA+1:z*NumEech4SA,:);                       
                    try
                        DiagAddRelAll = 1-AddRelSel((z-1)*NumEech4SA+1:z*NumEech4SA,:) .* AddRelSel((z-1)*NumEech4SA+1:z*NumEech4SA,:);                           
                    catch
                        DiagAddRelAll = [];
                    end
                end
                DiagAddEnh = sqrt(sum(DiagAddEnhAll.^2,1)./NumEech4SA);
                try
                    DiagAddRel = sqrt(sum(DiagAddRelAll.^2,1)./NumEech4SA);                          
                catch
                    DiagAddRel = [];
                end
                if model.Step == 0  
                    AddFeaEnhPDID = AddFeaEnhWei * diag(DiagAddEnh) * AddEnhPDD;
                    model.FeaPD{z} = AddFeaPDD + AddFeaEnhPDID;
                    model.AllPD{z} = [model.FeaPD{z};AddEnhPDD];
                else
                    OriFeaPD = model.FeaPD{z} + OriFeaAddEnhWei * diag(DiagAddEnh) * AddEnhPDD; 
                    AddFeaEnhPDID = AddFeaEnhWei * diag(DiagAddEnh) * AddEnhPDD;
                    AddFeaRelPDID = AddFeaRelWei * diag(DiagAddRel) * AddRelPDD;
                    AddFeaPD = AddFeaPDD + AddFeaEnhPDID + AddFeaRelPDID;
                    model.FeaPD{z} = [OriFeaPD;AddFeaPD];
                    for k = 1:model.Step-1
                        model.AllPD{z}(NumFeaOri+NumEnhOri+(NumAddFea+NumAddRel+NumAddEnh)*(k-1)+1:NumFeaOri+NumEnhOri+(NumAddFea+NumAddRel+NumAddEnh)*(k-1)+NumAddFea,:)...
                      = model.FeaPD{z}(NumFeaOri+NumAddFea*(k-1)+1:NumFeaOri+NumAddFea*k,:);
                    end                    
                    model.AllPD{z} = [model.AllPD{z} ; AddFeaPD ; AddRelPDD ; AddEnhPDD];
                end
                % import nodes to z-th class
                [~,Max_index] = max(model.AllPD{z},[],2);
                Max_index_{z}=find(Max_index==z); 
                AllPD_z = model.AllPD{z}(:,z) ;

%                 %% method of interseciton line
                [row_descend,index_temp] = sort(AllPD_z,"descend");
                row_descend_line = sort(linspace(min(row_descend),max(row_descend),length(model.Beta(:,1))),'descend'); 
                [~,ban_index] = min(abs(row_descend_line(2:end-1) - row_descend(2:end-1)'));

                %% method of tangent line
%                 [row_descend,index_temp] = sort(abs(AllPD_z),"descend");
%                 abs_index = 1:length(row_descend);
%                 [~,ban_index] = min(sqrt(row_descend'.^2+abs_index.^2));


                if row_descend(ban_index)>0
                else
                    [~,ban_index] = min(row_descend(row_descend>0));
                end
                Selected_row = index_temp(1:ban_index);                
                SelectNeurons = intersect(Selected_row,Max_index_{z});  
              
                % selec
                if length(SelectNeurons) < ceil(length(row_descend)*ThetaSel)
                    SelectNeruonSet = SelectNeruonSet;
                else
                    SelectNeruonSet = union(SelectNeruonSet,SelectNeurons);
                end
            end
            model.BanNodes = setdiff(index_temp,SelectNeruonSet);                      
        end
    end %methodq
end %class
         
        

         
        
