clc;
clear;
close all;

syms theta
a=1;
x=a*theta*cos(theta+pi);
y=a*theta*sin(theta+pi);
fplot(x,y,[0,pi*2*5],'LineWidth',1.5,'Color','b');
grid on
axis square
hold on
x=a*theta*cos(theta);
y=a*theta*sin(theta);
fplot(x,y,[0,pi*2*5],'LineWidth',1.5,'Color','r');