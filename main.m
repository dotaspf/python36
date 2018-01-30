 clc;clear;
 %默认训练集和测试集中的cover和 stego的比例是1:1
 images={'coolpad_pro2','huawei_6x','huawei_MT7','huawei_MT9','huawei_P9','huawei_V8','iphone_7','iphone_7p','meilan_note6',...
    'oppo_A37m','oppo_R9','oppo_R9p','samsung_C7','samsung_S7edge','sony_Z4','vivo_X6plus','vivo_X6s','vivo_X9','xiaomi_5','xiaomi_6'};
TST_ran=zeros(1,1); TST_TC=zeros(1,1); OOB_ran=zeros(1,1);T_TC=zeros(1,1);T_ran=zeros(1,1);OOB_TC=zeros(1,1);
acc6=zeros(1,1);
parameter.testname='vivo_X3L';
parameter.format='PGM';
parameter.fea='SPAM';
parameter.stego='LSBM';
parameter.n=100; 
parameter.p=0.2;
TCcal=0;%1为开0为关
parameter.classification='FLD';
%% 提取TC相似度部分
load(['.\TC_fea\suspicious\',parameter.format,'\',parameter.testname,'_cover']);
load(['.\TC_fea\suspicious\',parameter.format,'\',parameter.testname,'_',parameter.stego]);
testTC_cover=eval([parameter.testname,'_cover']);testTC_stego=eval([parameter.testname,'_',parameter.stego]);
lengthtestc=length(testTC_cover(:,1));lengthtests=length(testTC_stego(:,1));


if TCcal==1
[trainTC_cover,trainTC_stego] = loadTC(images,parameter);

testnum{1}=1:lengthtestc;testnum{2}=1:lengthtests;%测试集的选取，可以是一个随机数集合
lengthtrainc=length(trainTC_cover(:,1));lengthtrains=length(trainTC_stego(:,1));
allnum=lengthtrainc;

[covernum,stegonum]=TCselect4(trainTC_cover,trainTC_stego,testTC_cover,testTC_stego,testnum,allnum);
save([parameter.testname,'_num'],covernum,stegonum);
end

%%
load vivo_X3L_LSBM_num

[trainc,trains,testc,tests]= loadfea(images,parameter);
temp=1;

for sumtest=1000;
    parameter.begin=1;
    parameter.num=3;%决定了是选多个还是一个
    parameter.sumtestc=sumtest;
    parameter.sumtests=sumtest;
    sumtrain=(parameter.sumtestc+parameter.sumtests)*parameter.num;
        for j=1:25 
            parameter.sample_testc=sampling(1:length(covernum{1,1}(:,1)),parameter.sumtestc);
            parameter.sample_tests=sampling(1:length(stegonum{1,1}(:,1)),parameter.sumtests);
            parameter.sample_trainc=sampling(1:length(covernum{1,1}(1,3:end)),sumtrain);
            parameter.sample_trains=sampling(1:length(stegonum{1,1}(1,3:end)),sumtrain);
            covernum1{1,1}=covernum{1,1}(parameter.sample_testc,:);
            covernum1{2,1}=covernum{2,1}(parameter.sample_testc,:);
            stegonum1{1,1}=stegonum{1,1}(parameter.sample_tests,:);
            stegonum1{2,1}=stegonum{2,1}(parameter.sample_tests,:);
            %将测试集的TC值传到函数里面去，以便备用
            parameter.test_TCc=testTC_cover(parameter.sample_testc,:);
            parameter.test_TCs=testTC_cover(parameter.sample_tests,:);
            [TNTc,TNTs,TSTc,TSTs,index]=xuantu(trainc,trains,testc,tests,parameter,covernum1,stegonum1);
            %% 分类器选择
            % if strcmp(parameter.classification,'FLD')
            % addpath('.\classifier\FLD_ensemble\random');
            % [acc1,OOB1,T1]=random_classifier(TNTc,TNTs,TSTc,TSTs);
            % addpath('.\classifier\FLD_ensemble\TC_select');
            % [acc2,OOB2,T2]=ensemble_classifier(TNTc,TNTs,TSTc,TSTs,index);
            % end
            if strcmp(parameter.classification,'FLD')
                addpath('.\classifier\FLD_ensemble\random');
                [acc1,OOB1,T]=random_classifier(trainc,trains,TSTc,TSTs);
                addpath('.\classifier\FLD_ensemble\TC_select');
                [acc2,OOB2]=ensemble_classifier(TNTc,TNTs,TSTc,TSTs,index);
                total_random_oob_error(j) = OOB1;
                total_oob_error(j) = OOB2;
                total_error_R1(j) = acc1;
                total_error_R2(j) = acc2;
                    if j == 25
                        mean_rand_oob = mean(total_random_oob_error);
                        mean_rand_test_e = mean(total_error_R1);
                        mean_tc_oob = mean(total_oob_error);
                        mean_tc_test_e = mean(total_error_R2);

                    %     variance_rand_oob = sum((total_random_oob_error - mean_rand_oob).^2,1);
                    %     variance_test_e = sum((total_error_R1 - mean_rand_test_e).^2,1);
                    %     variance_tc_oob = sum((total_oob_error - mean_tc_oob).^2,1);
                    %     variance_tc_test_e = sum((total_error_R2 - mean_tc_test_e).^2,1);

                        fprintf('The Total Result!!!!!!!!!!!!!!!\n');
                        fprintf('R1 Testing OOB1: %.4f\t',mean_rand_oob);
                        fprintf('R1 Testing error1: %.4f\n',mean_rand_test_e);
                        fprintf('R2 Testing OOB2: %.4f\t',mean_tc_oob);
                        fprintf('R2 Testing error2: %.4f\n',mean_tc_test_e);
                        break;
                    end
                end
            if strcmp(parameter.classification,'SVM')
            addpath('.\classifier\SVM');
            acc1=TCrandom(trainc,trains,testc,tests,parameter);
            acc2=SVM(TNTc(:,2:end),TNTs(:,2:end),TSTc,TSTs); 
            end
        end
        temp=temp+1;
end
save acc_iphone_7 TST_ran TST_TC 

