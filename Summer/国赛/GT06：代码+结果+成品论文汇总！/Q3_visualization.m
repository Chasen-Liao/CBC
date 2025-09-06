% 参数
syms i r rs rp
N=15;           % 透射光研究条数
lambda=500;     % 波长
n=10;           % 折射率
h=25000;        % 平板厚度
Ai=1;           % 入射光振幅
a=25*pi/180;         % 入射光振幅方位角
theta_max=pi/3; % 像方视场角
delta_theta=0.001;  % 精度
Asi=Ai*sin(a);  % 入射s波振幅
Api=Ai*cos(a);  % 入射p波振幅

P=@(rs,rp)(rs.*sin(a)).^2+(rp.*cos(a)).^2;      % 反射比
Rs=@(i,r)-sin(i-r)./sin(i+r);                   % 菲涅尔公式
Rp=@(i,r)tan(i-r)./tan(i+r);
Ts=@(i,r)2*sin(r).*cos(i)./sin(i+r);
Tp=@(i,r)2*sin(r).*cos(i)./sin(i+r)./cos(i-r);

i1=(-theta_max:delta_theta:theta_max)';
r1=asin(sin(i1)/n);
delta=4*pi/lambda*n*h*cos(r1);

% 分sp波算
As=Asi*Ts(i1,r1).*Ts(r1,i1);
Ap=Api*Tp(i1,r1).*Tp(r1,i1);
Ast=As;
Apt=Ap;
for i=1:N-1
    As=As.*Rs(r1,i1).^2.*exp(1i*delta);
    Ap=Ap.*Rp(r1,i1).^2.*exp(1i*delta);
    Ast=Ast+As;
    Apt=Apt+Ap;
end
It_1=abs(Ast).^2+abs(Apt).^2;

% 各级透射光振幅方向
ii=pi/4;
r=asin(sin(i)/n);
As=Asi*Ts(ii,r).*Ts(r,ii);
Ap=Api*Tp(ii,r).*Tp(r,ii);
sols=As*[zeros(1,N-1);ones(1,N-1)];
solp=Ap*[zeros(1,N-1);ones(1,N-1)];
for i=2:N
    sols(2,i)=sols(2,1)*Rs(ii,r)^(2*i-2);
    solp(2,i)=solp(2,1)*Rp(ii,r)^(2*i-2);
end

% 教材
rs=Rs(i1,r1);
rp=Rp(i1,r1);
p=P(rs,rp);
It_2=(1-p).^2./((1-p).^2+4*p.*sin(delta/2).^2)*Ai^2;

% 绘图
figure(1)
[X,I]=meshgrid(i1,It_1);
Y=X';
surf(X,Y,I)
axis equal
colormap hot
view(90,90)
shading interp

figure(2)
subplot(1,2,1)
plot(i1,It_1,'b',i1,It_2,'r','linewidth',1.5)
legend('分sp波','不区分sp波')
xlabel('干涉角\theta','fontname','kaiti','fontsize',20)
ylabel('光强I','fontname','kaiti','fontsize',20)
title('两种计算结果对比','fontname','kaiti','fontsize',24)
grid on

subplot(1,2,2)
% num=cell(1,N-1);
% for i=2:N
%     num{i-1}=num2str(i);
% end
% num=char(num);
plot(sols,solp,'linewidth',1.5)
legend('1级','2级','3级','4级')
xlabel('s波振幅','fontname','kaiti','fontsize',20)
ylabel('p波振幅','fontname','kaiti','fontsize',20)
title('i=10°时各级透射光总振幅方向分析','fontname','kaiti','fontsize',24)
axis equal
grid on