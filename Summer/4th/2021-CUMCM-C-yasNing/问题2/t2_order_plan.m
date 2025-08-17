%% 根据供货商的供货能力初始化订货方案
load('..\predict_ABC.mat');
load('..\prob.mat');
%选定30家供货商
suppliers = xlsread('..\问题1\问题1.xlsx', '供应商重要性', 'A2:A31');
supply_ability = prob(:, 2);
class = prob(:, 3);
A = [];
B = [];
C = [];
total_A = 0;
total_B = 0;
total_C = 0;
for i = 1:30
    supplier = suppliers(i);
    if class(supplier) == 1.2
        A = [A, supplier];
        total_A = total_A + supply_ability(supplier);
    elseif class(supplier) == 1.1
        B = [B, supplier];
        total_B = total_B + supply_ability(supplier);
    elseif class(supplier) == 1
        C = [C, supplier];
        total_C = total_C + supply_ability(supplier);
    end
end

ability = zeros(402, 1);
order_plan = zeros(402, 24);
for i = 1:402
    if ismember(i, A)
        ability(i) = supply_ability(i) / total_A;
        for j = 1:24
            order_plan(i, j) = round(ability(i) * predict_ABC(j, 1));
        end
    elseif ismember(i, B)
        ability(i) = supply_ability(i) / total_B;
        for j = 1:24
            order_plan(i, j) = round(ability(i) * predict_ABC(j, 2));
        end
    elseif ismember(i, C) 
        ability(i) = supply_ability(i) / total_C;
        for j = 1:24
            order_plan(i, j) = round(ability(i) * predict_ABC(j, 3));
        end
    end
end

%% 模拟退火算法寻找最经济的订货方案
% 参数初始化
T0 = 1000;   % 初始温度
T = T0; % 迭代中温度会发生改变，第一次迭代时温度就是T0
maxOutTimes = 500;  % 外循环，最大迭代次数
maxInTimes = 100;  % 内循环，每个温度下的迭代次数
alfa = 0.98;  % 温度衰减系数
x_new = zeros(402, 24);
%计算订购成本, 目标函数值
[money, y0] = compute_money(order_plan, class);

%% 定义一些保存中间过程的量，方便输出结果和画图
[ys, min_y] = compute_money(order_plan, class);     
MINY = zeros(maxOutTimes, 1); 
temp = zeros(maxOutTimes, 1);

%% 模拟退火过程
% 外循环
for iter = 1 : maxOutTimes  
    % 内循环，在每个温度下开始迭代
    for i = 1 : maxInTimes 
        %% 随机产生新解,对新解进行边界约束
        for j = 1:402
            for k = 1:24
                if order_plan(j, k) ~= 0
                    x_new(j, k) = round(order_plan(j, k) * (1 + (rand() - 0.5) * 2 * 0.3));
                end
            end
        end
        
        %% 记录新解，并计算新解的函数值，保存最优解
        % 将调整后的x_new赋值给新解x1
        x1 = x_new;
         % 计算新解的函数值
        [ys1, y1] = compute_money(x1, class);
        % 如果新解函数值小于当前解的函数值
        if y1 < y0    
            % 更新当前解为新解
            x0 = x1; 
            y0 = y1;
            
        %%根据Metropolis准则计算一个概率
        else
            p = exp(-(y1 - y0) / T); 
            % 生成一个随机数和这个概率比较，如果该随机数小于这个概率
            if rand(1) < p
                % 更新当前解为新解
                x0 = x1; 
                y0 = y1;
            end
        end
        
        % 判断是否要更新找到的最佳的解
        % 如果当前解更好，则对其进行更新
        if y0 < min_y  
            % 更新最小的y
            min_y = y0;  
             % 更新找到的最好的解x
            best_x = x0; 
        end
    end
    
    % 记录本次外循环结束后找到的最优解
    MINY(iter) = min_y; 
    % 温度下降
    T = alfa * T;   
    temp(iter) = T;
end

disp('此时最优值是：'); 
disp(min_y);
xlswrite('..\附件A 订购方案数据结果.xlsx', best_x, '问题2的订购方案结果', 'B7:Y408');

%% 画出每次迭代后找到的最小值的图形
fig = figure;
X = 1:maxOutTimes;
yyaxis left
plot(X, MINY);
ylabel('目标函数值');
yyaxis right
plot(X, temp);
ylabel('温度');
xlabel('迭代次数');
title('模拟退火算法寻找最优订购方案');
legend('目标函数', '温度');  %加注图例

