classdef FPD_SA_Online
    properties
        Name = 'Off-line Broad Learning System';

    end
    %% Functions and algorithm
    methods    (Static = true)              
        function model = SA(model,SelTrainA,NumEech4SA,sigfun,mode)
            NumFeaOri = model.NumPerWin * model.NumWindow;
            NumEnhOri = model.NumEnhance; 
            NumAddFea = model.NumAddFea;
            NumAddRel = model.NumAddRel;
            NumAddEnh = model.NumAddEnh;
            NumClass = length(model.Beta(1,:));
            if model.AddNodeStep == 0      
                AddFeaPDD = model.Beta(1:NumFeaOri,:);
                AddEnhPDD = model.Beta(NumFeaOri + 1:NumFeaOri+NumEnhOri,:); 
                AddEnhSel = SelTrainA(:,NumFeaOri + 1:NumFeaOri+NumEnhOri);
                AddRelSel = [];
                AddFeaEnhWei = model.FeaEnhWei(1:NumFeaOri,:);
                OriFeaAddEnhWei = [];
                AddFeaRelWei = [];                
                AddRelPDD = [];              
            elseif strcmp(mode,'AN')                
                AddFeaPDD = model.Beta(end - (NumAddFea+NumAddEnh+NumAddRel) + 1:end - (NumAddEnh+NumAddRel),:);
                AddEnhSel = SelTrainA(:,end-NumAddEnh+1:end); 
                AddRelSel = SelTrainA(:,end-NumAddRel-NumAddEnh+1:end-NumAddEnh);  
                AddFeaEnhWei = model.AllFeaAddEnhWei{model.AddNodeStep}(end-NumAddFea:end-1,:);
                OriFeaAddEnhWei = model.AllFeaAddEnhWei{model.AddNodeStep}(1:end-NumAddFea-1,:);
                AddFeaRelWei = model.AddFeaRelWei{model.AddNodeStep}(1:NumAddFea,:);                
                AddRelPDD = model.Beta(end - NumAddRel - NumAddEnh+1:end -NumAddEnh,:);
                AddEnhPDD = model.Beta(end-NumAddEnh+1:end,:); 
            else
                AddDataOriEnhPDD = model.Beta(NumFeaOri+1:NumFeaOri+NumEnhOri,:);
                AddDataOriEnhSel = SelTrainA(:,NumFeaOri+1:NumFeaOri+NumEnhOri);
            end 
            SelectNeruonSet = [];
            if strcmp(mode,'AN') || model.AddNodeStep == 0 
                for z = 1:NumClass  
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
                    if model.AddNodeStep == 0  
                        AddFeaEnhPDID = AddFeaEnhWei * diag(DiagAddEnh) * AddEnhPDD;
                        model.FeaPD{z} = AddFeaPDD + AddFeaEnhPDID;
                        model.AllPD{z} = [model.FeaPD{z};AddEnhPDD];
                    else
                        OriFeaPD = model.FeaPD{z} + OriFeaAddEnhWei * diag(DiagAddEnh) * AddEnhPDD; 
                        AddFeaEnhPDID = AddFeaEnhWei * diag(DiagAddEnh) * AddEnhPDD;
                        AddFeaRelPDID = AddFeaRelWei * diag(DiagAddRel) * AddRelPDD;
                        AddFeaPD = AddFeaPDD + AddFeaEnhPDID + AddFeaRelPDID;
                        model.FeaPD{z} = [OriFeaPD;AddFeaPD];
                        for k = 1:model.AddNodeStep-1
                            model.AllPD{z}(NumFeaOri+NumEnhOri+(NumAddFea+NumAddRel+NumAddRel)*(k-1)+1:NumFeaOri+NumEnhOri+(NumAddFea+NumAddRel+NumAddRel)*(k-1)+NumAddFea,:)...
                          = model.FeaPD{z}(NumFeaOri+NumAddFea*(k-1)+1:NumFeaOri+NumAddFea*k,:);
                        end                    
                        model.AllPD{z} = [model.AllPD{z} ; AddFeaPD ; AddRelPDD ; AddEnhPDD];                        
                    end
                    [~,Max_index] = max(model.AllPD{z},[],2);
                    Max_index_{z}=find(Max_index==z); 
                    AllPD_z = model.AllPD{z}(:,z) ;   
                    [row_descend,index_temp] = sort(abs(AllPD_z),"descend");
                    [row_descend,index_temp] = sort(AllPD_z,"descend");
                    row_descend_line = sort(linspace(min(row_descend),max(row_descend),length(model.Beta(:,1))),'descend'); 
                    [~,ban_index] = min(abs(row_descend_line(2:end-1) - row_descend(2:end-1)'));
                    

                %% method of tangent line                    
%                     abs_index = 1:length(row_descend);
%                     [~,ban_index] = min(sqrt(row_descend'.^2+abs_index.^2));


                    if row_descend(ban_index)>0
                    else
                        [~,ban_index] = min(row_descend(row_descend>0));
                    end
                    Selected_row = index_temp(1:ban_index);                
                    SelectNeurons = intersect(Selected_row,Max_index_{z});
                    SelectNeruonSet = union(SelectNeruonSet,SelectNeurons);
                end
            else
               for z = 1:NumClass  
                   if strcmp(sigfun,'logsig')
                        DiagOriEnh = AddDataOriEnhSel((z-1)*NumEech4SA+1:z*NumEech4SA,:) .*(1-AddDataOriEnhSel((z-1)*NumEech4SA+1:z*NumEech4SA,:));  
                   else
                        DiagOriEnh = 1-AddDataOriEnhSel((z-1)*NumEech4SA+1:z*NumEech4SA,:) .* AddDataOriEnhSel((z-1)*NumEech4SA+1:z*NumEech4SA,:);  
                   end
                   DiagOriEnh = sqrt(sum(DiagOriEnh.^2,1)./NumEech4SA);
                   OriFeaOriEnhPDID = model.FeaEnhWei(1:NumFeaOri,:) * diag(DiagOriEnh) * AddDataOriEnhPDD;
                   model.FeaPD{z}(1:NumFeaOri,:) = model.FeaPD{z}(1:NumFeaOri,:) + OriFeaOriEnhPDID;
                   for i = 1:model.AddNodeStep
                        AddDataAddRelSel = SelTrainA(:,NumFeaOri+NumEnhOri+(NumAddFea+NumAddRel+NumAddEnh)*(i-1)+NumAddFea+1:NumFeaOri+NumEnhOri+(NumAddFea+NumAddRel+NumAddEnh)*(i-1)+NumAddFea+NumAddRel);
                        AddDataAddEnhSel = SelTrainA(:,NumFeaOri+NumEnhOri+(NumAddFea+NumAddRel+NumAddEnh)*(i-1)+NumAddFea+NumAddRel+1:NumFeaOri+NumEnhOri+(NumAddFea+NumAddRel+NumAddEnh)*(i-1)+NumAddFea+NumAddRel+NumAddEnh);
                        if strcmp(sigfun,'logsig')
                            DiagAddStepRel = AddDataAddRelSel((z-1)*NumEech4SA+1:z*NumEech4SA,:) .*(1-AddDataAddRelSel((z-1)*NumEech4SA+1:z*NumEech4SA,:));                            
                            DiagAddStepEnh = AddDataAddEnhSel((z-1)*NumEech4SA+1:z*NumEech4SA,:) .*(1-AddDataAddEnhSel((z-1)*NumEech4SA+1:z*NumEech4SA,:));  
                        else
                            DiagAddStepRel = 1-AddDataAddRelSel((z-1)*NumEech4SA+1:z*NumEech4SA,:) .* AddDataAddRelSel((z-1)*NumEech4SA+1:z*NumEech4SA,:);  
                            DiagAddStepEnh = 1-AddDataAddEnhSel((z-1)*NumEech4SA+1:z*NumEech4SA,:) .* AddDataAddEnhSel((z-1)*NumEech4SA+1:z*NumEech4SA,:);  
                        end
                        DiagAddRel = sqrt(sum(DiagAddStepRel.^2,1)./NumEech4SA); 
                        DiagAddEnh = sqrt(sum(DiagAddStepEnh.^2,1)./NumEech4SA);
                        AddFeaRelWei = model.AddFeaRelWei{i}(1:NumAddFea,:);
                        OriFeaAddEnhWei = model.AllFeaAddEnhWei{i}(1:NumFeaOri+i*NumAddFea,:);
                        AddRelPDD = model.Beta(NumFeaOri+NumEnhOri+(NumAddFea+NumAddRel+NumAddEnh)*(i-1)+NumAddFea+1:NumFeaOri+NumEnhOri+(NumAddFea+NumAddRel+NumAddEnh)*(i-1)+NumAddFea+NumAddRel,:);
                        AddEnhPDD = model.Beta(NumFeaOri+NumEnhOri+(NumAddFea+NumAddRel+NumAddEnh)*(i-1)+NumAddFea+NumAddRel+1:NumFeaOri+NumEnhOri+(NumAddFea+NumAddRel+NumAddEnh)*(i-1)+NumAddFea+NumAddRel+NumAddEnh,:); 
                        AddFeaRelPDID = AddFeaRelWei * diag(DiagAddRel) * AddRelPDD;
                        OriFeaAddEnhPDID = OriFeaAddEnhWei * diag(DiagAddEnh) * AddEnhPDD;
                        model.FeaPD{z}(1:NumFeaOri+i*NumAddFea,:) = model.FeaPD{z}(1:NumFeaOri+i*NumAddFea,:) + OriFeaAddEnhPDID;
                        model.FeaPD{z}(NumFeaOri+(i-1)*NumAddFea + 1:NumFeaOri+i*NumAddFea,:) = model.FeaPD{z}(NumFeaOri+(i-1)*NumAddFea + 1:NumFeaOri+i*NumAddFea,:) + AddFeaRelPDID;
                   end
                   model.AllPD{z}(1:NumFeaOri,:) = model.FeaPD{z}(1:NumFeaOri,:);
                   for k = 1:model.AddNodeStep
                            model.AllPD{z}(NumFeaOri+NumEnhOri+(NumAddFea+NumAddRel+NumAddRel)*(k-1)+1:NumFeaOri+NumEnhOri+(NumAddFea+NumAddRel+NumAddRel)*(k-1)+NumAddFea,:)...
                          = model.FeaPD{z}(NumFeaOri+NumAddFea*(k-1)+1:NumFeaOri+NumAddFea*k,:);
                   end 
                    [~,Max_index] = max(model.AllPD{z},[],2);
                    Max_index_{z}=find(Max_index==z); 
                    AllPD_z = model.AllPD{z}(:,z) ;   
                    [row_descend,index_temp] = sort(AllPD_z,"descend");                
                    row_descend_line = sort(linspace(min(row_descend),max(row_descend),length(model.Beta(:,1))),'descend'); 
                    [~,ban_index] = min(abs(row_descend_line(2:end-1) - row_descend(2:end-1)'));
                    

                %% method of tangent line
%                     abs_index = 1:length(row_descend);
%                     [~,ban_index] = min(sqrt(row_descend'.^2+abs_index.^2));



                    if row_descend(ban_index)>0
                    else
                        [~,ban_index] = min(row_descend(row_descend>0));
                    end
                    Selected_row = index_temp(1:ban_index);                
                    SelectNeurons = intersect(Selected_row,Max_index_{z});
                    SelectNeruonSet = union(SelectNeruonSet,SelectNeurons);
               end
            end
            model.BanNodes = setdiff(index_temp,SelectNeruonSet);                      
        end
    end %methodq
end %class
         
        

         
        
