%%根据最优化得到的订货方案，寻找最优转运方案
%损耗=本周本供应商供货量*损耗率，数值为0表示没有运送
order_plan = xlsread('..\附件A 订购方案数据结果.xlsx', '问题2的订购方案结果', 'B7:Y408');
load('..\predict_waste.mat');
load('..\prob.mat');
class = prob(:, 3);

%% 模拟退火算法寻找损耗最少的转运方案
T0 = 1000;
T = T0;
maxOutTimes = 500;  % 外循环
maxInTimes = 100;  % 内循环
alfa = 0.98; 
x0 = Init_trans(order_plan, predict_waste);
init = x0;
%计算损耗量
W0 = compute_waste(x0, predict_waste, class);

%% 定义一些保存中间过程的量
min_W = W0;     
MINY = zeros(maxOutTimes, 1);
temp = zeros(maxOutTimes, 1);

%% 模拟退火过程
% 外循环
for iter = 1 : maxOutTimes  
    % 内循环
    for i = 1 : maxInTimes 
        %% 随机产生新解
        x1 = Init_trans(order_plan, predict_waste);
        
        %% 记录新解，并计算新解的函数值，保存最优解
        % 计算新解的函数值
        W1 = compute_waste(x1, predict_waste, class);
        % 如果新解函数值小于当前解的函数值
        if W1 < W0    
            % 更新当前解为新解
            x0 = x1; 
            W0 = W1;
            
        %%根据Metropolis准则计算一个概率
        else
            p = exp(-(W1 - W0) / T); 
            if rand(1) < p
                % 更新当前解为新解
                x0 = x1; 
                W0 = W1;
            end
        end
        
        % 判断是否要更新找到的最佳的解
        % 如果当前解更好，则对其进行更新
        if W0 < min_W  
            min_W = W0;  
            best_x = x0; 
        end
    end
    
    % 记录本次外循环结束后找到的最优解
    MINY(iter) = min_W; 
    % 温度下降
    T = alfa * T; 
    temp(iter) = T;
end

disp('此时最优值是：'); 
disp(min_W);
xlswrite('..\附件B 转运方案数据结果.xlsx', best_x, '问题2的转运方案结果', 'B7:GK408');

%% 画出每次迭代后找到的最小值的图形
fig = figure;
X = 1:maxOutTimes;
yyaxis left
plot(X, MINY);
ylabel('损耗');
yyaxis right
plot(X, temp);
ylabel('温度');
xlabel('迭代次数');
title('模拟退火算法寻找最优转运方案');
legend('损耗', '温度');  %加注图例