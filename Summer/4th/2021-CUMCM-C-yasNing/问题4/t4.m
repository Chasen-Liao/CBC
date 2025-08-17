load('..\predict_402.mat');
load('..\predict_waste.mat');
load('..\prob.mat');
class = prob(:, 3);

%% 模拟退火算法寻找产能最大
T0 = 1000;
T = T0;
maxOutTimes = 500;  % 外循环
maxInTimes = 100;  % 内循环
alfa = 0.98; 
%随机产生初始解
[x10, x20, capacity0] = t4_init(predict_402, predict_waste, class);
y0 = min(capacity0);

%% 定义一些保存中间过程的量
max_y = y0;   
MAXY = zeros(maxOutTimes, 1);
temp = zeros(maxOutTimes, 1);

%% 模拟退火过程
% 外循环
for iter = 1 : maxOutTimes  
    % 内循环
    for i = 1 : maxInTimes 
        %% 随机产生新解
        [x1, x2, capacity] = t4_init(predict_402, predict_waste, class);
        
        %% 记录新解，并计算新解的函数值，保存最优解
        y1 = min(capacity);
        % 如果新解函数值大于当前解的函数值
        if y1 > y0    
            % 更新当前解为新解
            x10 = x1; 
            x20 = x2;
            y0 = y1;
            capacity0 = capacity;
            
        %%根据Metropolis准则计算一个概率
        else
            p = exp(-(y0 - y1) / T); 
            if rand(1) < p
                % 更新当前解为新解
                x10 = x1; 
                x20 = x2;
                y0 = y1;
                capacity0 = capacity;
            end
        end
        
        % 判断是否要更新找到的最佳的解
        if y0 > max_y  
            max_y = y0;  
            best_x1 = x10; 
            best_x2 = x20; 
            best_c = capacity;
        end
    end
    
    % 记录本次外循环结束后找到的最优解
    MAXY(iter) = max_y; 
    T = alfa * T; 
    temp(iter) = T;
end

disp('该企业每周的产能为：'); 
disp(max_y);
xlswrite('..\附件A 订购方案数据结果.xlsx', best_x1, '问题4的订购方案结果', 'B7:Y408');
xlswrite('..\附件B 转运方案数据结果.xlsx', best_x2, '问题4的转运方案结果', 'B7:GK408');

%% 画出每次迭代后找到的最小值的图形
fig = figure;
X = 1:maxOutTimes;
yyaxis left
plot(X, MAXY);
ylabel('产能');
yyaxis right
plot(X, temp);
ylabel('温度');
xlabel('迭代次数');
title('模拟退火算法寻找企业最优产能');
legend('产能', '温度');