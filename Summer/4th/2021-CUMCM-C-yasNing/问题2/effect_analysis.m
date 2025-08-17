%% 对订购方案的实施效果分析
order_plan = xlsread('..\附件A 订购方案数据结果.xlsx', '问题2的订购方案结果', 'B7:Y408');
%未初始化时
% load('order_plan.mat');
%库存产能
capacity = 0;
satisfy_0 = 0;
satisfy_1 = 0;
satisfy_2 = 0;
satisfy_3 = 0;
load('..\prob.mat');
% 供货商的材料种类
class = prob(:, 3);
for i = 1:24
    for j = 1:402
        if class(j) == 1.2
            capacity = order_plan(j, i) / 0.6 + capacity;
        elseif class(j) == 1.1
            capacity = order_plan(j, i) / 0.66 + capacity;
        elseif class(j) == 1
            capacity = order_plan(j, i) / 0.72 + capacity;
        end
    end
    if capacity >= 2.82 * 1e4 * 3
        satisfy_3 = satisfy_3 + 1;
    elseif capacity >= 2.82 * 1e4 * 2
        satisfy_2 = satisfy_2 + 1;
    elseif capacity >= 2.82 * 1e4
        satisfy_1 = satisfy_1 + 1;
    else
        satisfy_0 = satisfy_0 + 1;
    end
    capacity = capacity - 2.82 * 1e4;
end
disp('订购方案实施效果分析结果：');
disp('满足三周生产需求的百分比：'); 
disp(satisfy_3 / 24 * 100);
disp('满足两周生产需求的百分比：'); 
disp((satisfy_3 + satisfy_2) / 24 * 100);
disp('满足一周生产需求的百分比：'); 
disp((satisfy_1 + satisfy_3 + satisfy_2) / 24 * 100);
disp('不能满足一周生产需求的百分比：'); 
disp(satisfy_0 / 24 * 100);

%% 对转运方案的实施效果分析
trans_plan = xlsread('..\附件B 转运方案数据结果.xlsx', '问题2的转运方案结果', 'B7:GK408');
%未优化时
% load('trans_plan.mat');
load('..\predict_waste.mat');
capacity = 0;
satisfy_0 = 0;
satisfy_1 = 0;
satisfy_2 = 0;
satisfy_3 = 0;
for i = 1:24
    for j = 1:402
        wasteage = 0;
        for t = 1:8
            wasteage = wasteage + trans_plan(j, (i - 1) * 8 + t) * predict_waste(t, i) / 100;
        end
        if class(j) == 1.2
            capacity = (order_plan(j, i) - wasteage) / 0.6 + capacity;
        elseif class(j) == 1.1
            capacity = (order_plan(j, i) - wasteage) / 0.66 + capacity;
        elseif class(j) == 1
            capacity = (order_plan(j, i) - wasteage) / 0.72 + capacity;
        end
    end
    if capacity >= 2.82 * 1e4 * 3
        satisfy_3 = satisfy_3 + 1;
    elseif capacity >= 2.82 * 1e4 * 2
        satisfy_2 = satisfy_2 + 1;
    elseif capacity >= 2.82 * 1e4
        satisfy_1 = satisfy_1 + 1;
    else
        satisfy_0 = satisfy_0 + 1;
    end
    capacity = capacity - 2.82 * 1e4;
end
disp('转运方案实施效果分析结果：');
disp('满足三周生产需求的百分比：'); 
disp(satisfy_3 / 24 * 100);
disp('满足两周生产需求的百分比：'); 
disp((satisfy_3 + satisfy_2) / 24 * 100);
disp('满足一周生产需求的百分比：'); 
disp((satisfy_1 + satisfy_3 + satisfy_2) / 24 * 100);
disp('不能满足一周生产需求的百分比：'); 
disp(satisfy_0 / 24 * 100);
            
