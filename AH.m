clc
clear all 
close all


%% importing our data
data = importdata('Missouri_Omaha_dailyQcfs_1929_2018.asc');
data2=importdata('Platte_Louisville_dailyQcfs_1954_2018.asc');
% putting variable into vectors for Missouri
day   =data(:,1);
month =data(:,2);
year  =data(:,3);
Q1  =data(:,4);
%Converting ft^3/s to m^3/s
Q1 = Q1/35.314666667;
t_M=datetime(year, month, day);
% putting variable into vectors for Platte
day2   =data2(:,1);
month2 =data2(:,2);
year2  =data2(:,3);
Q2  =data2(:,4);

%Converting ft^3/s to m^3/s
Q2 = Q2/35.314666667;
t_P=datetime(year2, month2, day2);

%Looking for negative values and substituting with NaN
%Missouri river
i=1;
for i=1:length(Q1)
    if Q1(i)<0
        Q1(i)=NaN;
        n(i)=i;
        i=i+1;
    end
end

%Platte river
i=1;
for i=1:length(Q2)
    if Q2(i)<0        
        neg_val(i)=Q2(i);
        n(i)=i;
        Q2(i)=NaN;
        i=i+1;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% plot daily stream flow of Missouri %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(1);
plot(t_M,Q1)
xlabel('time [days]')
ylabel('Q [m^3/s]')
datetick('x','yyyy')
hold on
plot(t_P, Q2)
legend('Missouri River','Platte River')
title('Daily Discharge of Missouri and Platte River')
figure(2)
plot(t_M, Q1)
xlabel('time [days]')
ylabel('Q [m^3/s]')



figure(3)
plot(t_P, Q2, 'r')
xlabel('time [days]')
ylabel('Q [m^3/s]')
title('Platte daily discharge')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %% log of time series of Missouri %%%%%%
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(4)
plot(t_M, Q1)
set(gca, 'Yscale', 'log')
title('Missouri daily discharge - log scale')
xlabel('time [days]')
ylabel('Q [m^3/s]')

%% log of time series of Platte %%
figure(5)
plot(t_P, Q2)
set(gca, 'Yscale', 'log')
title('Platte daily discharge - log scale')
xlabel('time [days]')
ylabel('Q [m^3/s]')

%% figuring both in one plot %%
figure(6)
plot(t_M, Q1)
xlabel('time [days]')
ylabel('Q [m^3/s]')
set(gca, 'Yscale', 'log')
hold on 
plot(t_P, Q2)
legend('Missouri daily discharge','Platte daily discharge')
title('Missouri and Platte daily discharge - log scale')
hold off

figure(7)
plot(t_M, smooth(Q1,0.01,'moving'))
xlabel('time [days]')
ylabel('Q [m^3/s]')
set(gca, 'Yscale', 'log')
hold on 
plot(t_P, smooth(Q2,0.01,'moving'))
legend('Missouri daily discharge','Platte daily discharge')
title('Smoothed daily discharge timeseries - Moving average')

 
%% Missouri and platte pristine and current


yypt     = min(year2):max(year2);
yyp_cur  = min(year2):max(year2);


yymt     = min(year):max(year);
yym_cur  = 1954:2018;
yym_pris = 1929:1954;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% pristine both of them %
%%%%%%%%%%%%%%%%%%%%%%%%%%%

Qdy_Mp=zeros(365,length(yym_pris))*NaN;
for i=1:length(yym_pris)
    dummy_M = Q1(year == yym_pris(i));
    Qdy_Mp(:,i) = dummy_M(1:365);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%
% current both of them %
%%%%%%%%%%%%%%%%%%%%%%%%%%%
Qdy_Pc=zeros(365,length(yyp_cur))*NaN;
Qdy_Mc=zeros(365,length(yym_cur))*NaN;
for i=1:length(yyp_cur)
    dummy_Pc = Q2(year2 == yyp_cur(i));
    Qdy_Pc(:,i) = dummy_Pc(1:365);
end
for i=1:length(yym_cur)
    dummy_Mc = Q1(year == yym_cur(i));
    Qdy_Mc(:,i) = dummy_Mc(1:365);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% total for both %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Qdy_P=zeros(365,length(yypt))*NaN;
Qdy_M=zeros(365,length(yymt))*NaN;
for i=1:length(yypt)
    dummy_P = Q2(year2 == yypt(i));
    Qdy_P(:,i) = dummy_P(1:365);
end
for i=1:length(yymt)
    dummy_M = Q1(year == yymt(i));
    Qdy_M(:,i) = dummy_M(1:365);
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Mean annual discharge total %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

maQ_P=zeros(length(yypt), 1)*NaN;

for i=1:length(yypt)   
    maQ_P(i) = round(nanmean(Qdy_P(:,i)));
end

maQ_M=zeros(length(yymt), 1)*NaN;
for i=1:length(yymt)   
    maQ_M(i) = round(nanmean(Qdy_M(:,i)));
end

totmean_M=round(mean(maQ_M));
totmean_P=round(mean(maQ_P));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%calculating Mean annual discharge pristine for both %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

maQ_M_pr=zeros(length(yym_pris), 1)*NaN;
for i=1:length(yym_pris)
    maQ_M_pr(i) = round(nanmean(Qdy_Mp(:,i)));
end
totmean_M_pr=round(nanmean(maQ_M_pr));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%calculating Mean annual discharge current for both %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
maQ_M_cur=zeros(length(yym_cur), 1)*NaN;
for i=1:length(yym_cur)
    maQ_M_cur(i) = round(nanmean(Qdy_Mc(:,i)));
end
totmean_M_cur=round(nanmean(maQ_M_cur));

maQ_P_cur=zeros(length(yyp_cur), 1)*NaN;
for i=1:length(yyp_cur)
    maQ_P_cur(i) = round(nanmean(Qdy_Pc(:,i)));
end
totmean_P_cur=round(nanmean(maQ_P_cur));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% making plot ^___^ %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure(8)
% mean annual discharge missouri (current)
plot(yym_cur,maQ_M_cur,'b')
hold on
plot(yym_pris,maQ_M_pr,'r')
hold on
% total mean missouri
yline(totmean_M, '--g');
hold on
yline(totmean_M_cur, '--b');
hold on
yline(totmean_M_pr, '--r');
hold on
% mean annual discharge platte (current)
plot(yyp_cur,maQ_P_cur,'b')
hold on 
% total mean platte
yline(totmean_P_cur, '--b');

xlabel('Year')
ylabel('Annual discharge [m^3/s]')
title('Mean Annual Discharge Timeseries for current and pristine conditions')
legend('Missouri-Cur.','Missouri-Pris.','tot. mean Missouri','tot. mean Missouri for Cur. Cond.','Tot. Mean Missouri for Pris. Cond.','Platte-Current','tot. mean Platte for Cur. Cond.','Location','best','NumColumns',3)
axis([1929 2018 0 2800])


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%calculating the mean regime curve 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Total mean regime curve for Missouri
mm = 1:12;
Qmr_M = zeros(length(mm), 1);

for (i = 1:length(mm))
    dummy = Q1(month == mm(i));
    Qmr_M(i) = round(mean(dummy(~isnan(dummy))));    
end

%Total mean regime curve for Platte
mm = 1:12;
Qmr_P = zeros(length(mm), 1);

for (i = 1:length(mm))
    dummy2 = Q2(month2== mm(i));
    Qmr_P(i) = round(mean(dummy2(~isnan(dummy2))));    % mm/a
end

figure(9)
plot(mm,Qmr_M,'-b')
hold on
plot(mm,Qmr_P,'-r')
xlabel('month in year')
ylabel('Monthly Discharge')
title('Total mean regime curve')
legend('Missouri river','Platte river')
hold off



%Mean regime curve for pristine Missouri
mm = 1:12;
Qmr_Mpr = zeros(length(mm), 1);

for (i = 1:length(mm))
     dummy3 =Q1(month== mm(i) & year<1954 );
    Qmr_Mpr(i) = round(mean(dummy3(~isnan(dummy3))));    % mm/a
end


figure (10)
plot(mm,Qmr_Mpr,'-b')
hold on
xlabel('month in year')
ylabel('Monthly Discharge')
title('Mean regime curve for pristine conditions')

%Mean regime curve for current flow Missouri

mm = 1:12;
Qmr_Mcr = zeros(length(mm), 1);

for (i = 1:length(mm))
     dummy3 =Q1(month== mm(i) & year>=1954 );
    Qmr_Mcr(i) = round(mean(dummy3(~isnan(dummy3))));    % mm/a
end


figure (100)
plot(mm,Qmr_Mcr,'-b','LineWidth', 2)
hold on
plot(mm,Qmr_Mpr,'-r','LineWidth', 2)
hold on
xlabel('month in year')
ylabel('Monthly Discharge')
legend('Missouri-Cur.', 'Missouri-Prist')
title('Mean regime curve for cur. & pris. conditions of Missouri')
hold off

%Mean regime curve for current flow Platte

mm = 1:12;
Qmr_Pcr = zeros(length(mm), 1);

for (i = 1:length(mm))
     dummy3 =Q2(month2== mm(i) & year2>=1954);
    Qmr_Pcr(i) = round(mean(dummy3(~isnan(dummy3))));    % mm/a
end

figure (11)
plot(mm,Qmr_Mcr,'-b','LineWidth', 2)
hold on
plot(mm,Qmr_Pcr,'-r','LineWidth', 2)
xlabel('month in year')
ylabel('Monthly Discharge')
legend('Missouri river','Platte river')
title('Mean regime curve for current conditions')
hold off


%% Mean annual Flow Duration Curves for Missouri and Platte

%Pristine

aFDC_Mp=Qdy_Mp*NaN;

for(i=1:length(yym_pris))
    aFDC_Mp(:,i)=sort(Qdy_Mp(:,i),'descend');
end


maFDC_Mp = zeros(365, 1)*NaN;
for (j = 1:365)
  dummy = aFDC_Mp(j,:);
  
  maFDC_Mp(j) = mean(dummy(~isnan(dummy)));
end


figure(12)
hold on
plot(1:365, maFDC_Mp, 'r-', 'LineWidth', 2)
xlabel('days in year in which Q is exceeded')
ylabel('Q  (m^3/s)')
set(gca, 'YScale', 'log')
title('Mean annual FDC for Misouri and Platte/ pristine conditions')
legend('Missouri-Pris.','Location','best','NumColumns',3)
ylim([0 10^5])

%Current

aFDC_Mc=Qdy_Mc*NaN;
aFDC_Pc=Qdy_Pc*NaN;

for(i=1:length(yym_cur))
    aFDC_Mc(:,i)=sort(Qdy_Mc(:,i),'descend');
end

for(i=1:length(yyp_cur))
    aFDC_Pc(:,i)=sort(Qdy_Pc(:,i),'descend');
end

maFDC_Mc = zeros(365, 1)*NaN;
for (j = 1:365)
  dummy = aFDC_Mc(j,:);
  
  maFDC_Mc(j) = mean(dummy(~isnan(dummy)));
end

maFDC_Pc = zeros(365, 1)*NaN;
for (j = 1:365)
  dummy = aFDC_Pc(j,:);
  
  maFDC_Pc(j) = mean(dummy(~isnan(dummy)));
end

figure(13)
hold on
plot(1:365, maFDC_Mc, 'b-', 'LineWidth', 2)
plot(1:365, maFDC_Pc, 'r-', 'LineWidth', 2)
xlabel('days in year in which Q is exceeded')
 ylabel('Q  (m^3/s)')
set(gca, 'YScale', 'log')
title('Mean annual FDC for Misouri and Platte, Current Conditions')
legend('Missouri-Cur.', 'Platte-Cur.','Location','best','NumColumns',3)
ylim([0 10^5])
xlim([1 365])



figure(14)
hold on
plot(1:365, maFDC_Mc, 'b-', 'LineWidth', 2)
plot(1:365, maFDC_Mp, 'r-', 'LineWidth', 2)
xlabel('days in year in which Q is exceeded')
 ylabel('Q  (m^3/s)')
set(gca, 'YScale', 'log')
title('Mean annual FDC for Misouri')
legend('Missouri-Cur.', 'Missouri-Prist','Location','best','NumColumns',3)
ylim([0 10^5])
xlim([1 365])


figure(15)
hold on
plot(1:365, maFDC_Pc, 'r-', 'LineWidth', 1)
xlabel('days in year in which Q is exceeded')
 ylabel('Q  (m^3/s)')
set(gca, 'YScale', 'log')
title('Mean annual FDC for Platte')
legend('Platte-Cur.', 'Location','best','NumColumns',3)
ylim([0 10^5])


%%


Q_Mavg = Qmr_Mcr(2);
Q_Pavg = Qmr_P(2);


%% Miinimum of Feb for missouri
mm = 1:12;
Qm_Min = zeros(length(yym_cur), 1);

for (i = 1:length(yym_cur))
   dummy3 =Q1( year == yym_cur(i) & month==2  );
   Qm_Min(i) = (min(dummy3(~isnan(dummy3)))); 
end


mm = 1:12;
Qp_Min = zeros(length(yyp_cur), 1);

for (i = 1:length(yyp_cur))
   dummy3 =Q2( year2 == yyp_cur(i) & month2==2  );
   Qp_Min(i) = (min(dummy3(~isnan(dummy3)))); 
end

%% Min of feb

Q_Mlowfeb= min(Qm_Min)

Q_Plowfeb= min(Qp_Min)

figure (16)
plot(yym_cur, Qm_Min,'-b','LineWidth', 2)
xlabel('Years')
ylabel('Low Flows [m^3/s]')
title('February Low flows for Missouri river')
hold on
plot(yyp_cur, Qp_Min, '-r','LineWidth', 2)
legend('Missouri River','Platte River')
hold off

figure (18)
bar(Qp_Min)
sort(Qm_Min)
Range= max(Qm_Min)-min(Qm_Min);


[counts,bins] = hist(Qm_Min,12); %# get counts and bin locations
barh(bins,counts,1)

ord_Qm_Min = sort(Qm_Min, 'ascend');
%ord_maxQ_P = sort(maxQ_P, 'ascend');

n= length(yym_cur);
k = 1:length(yym_cur);
for i=1:length(yym_cur)    
    Fs(i) = i/(length(yym_cur)+1);
end

figure (19)
plot(Fs,ord_Qm_Min,'.-') %empirical distribution function
xlabel( 'sample values [m^3/s]')
ylabel( 'Weibull plotting postion') 


%for Gumbel distribution 
ui = -log( -log( Fs ) ); %Reduced variate

figure(20)
plot(ord_Qm_Min,ui, '*')
xlabel('sample values [m3/s]')
ylabel('reduced variate [-]')
title('Gumbel probability plot')% we cant say yet that are perfectly aligned INTERMEDIATE TYPE OF ALIGNMENT


%for exponential distribution

ui = -log(1 -Fs ); %Reduced variate
 

figure(21)
plot(ord_Qm_Min,ui, '*')
xlabel('sample values [m3/s]')
ylabel('reduced variate [-]')
title('Exponential probability plot') %We can say that it is a bit better than gumber but we need analytical expressions
%to justify the choice, we need statistical inference


%for normal distribution
mean_ord_Qm_Min=mean(ord_Qm_Min)
std_ord_Qm_Min=std(ord_Qm_Min)
ui= norminv (Fs,mean_ord_Qm_Min,std_ord_Qm_Min)



figure(22)
plot(ord_Qm_Min,ui, 'o', 'markersize', 12)

xlabel('sample values [m3/s]')
ylabel('reduced variate [-]')
title('Normal probability plot')
%set(gca,'fontsize', 20)


% lognormal distribution

y = log (ord_Qm_Min);
ui = norminv(Fs);


figure(23)
plot(y,ui, 'o', 'markersize', 12)

xlabel('log sample values [m3/s]')
ylabel('reduced variate [-]')
title(' Log Normal probability plot')


%% %-------------- Moments & L-Moments --------------
mu=mean(ord_Qm_Min);

%Gumbel
%------- MOMENTS -------
std1=std(ord_Qm_Min);
teta2_Gu=std1*sqrt(6)/pi;
gamma=0.5772;
teta1_Gu=mu-gamma*teta2_Gu;
%Method of L_moments
b0 = mu;
n=length(ord_Qm_Min);
w=((1:n)-1)/(n-1);

b1=sum(w'.*ord_Qm_Min)/n;  %moltiplication element by element w' transpose vector to be compliant with xi
w1=((1:n)-1).*((1:n)-2)/((n-1)*(n-2));
b2=sum( w1'.*ord_Qm_Min)/n;

l1=b0;
l2 = 2*b1 - b0;
l3 = 6*b2 - 6*b1 + b0;

teta2_Gu_L= l2/log(2);
teta1_Gu_L = l1-0.5772*teta2_Gu_L;

%exponential
teta_ex=mu;
%Method of L_moments
teta_ex_L=l1;

%Normal
teta1_Norm=mu;
teta2_Norm=std1;
%Method of L_moments
teta1_Norm_L=l1;
teta2_Norm_L=sqrt(pi)*l2;

% lognormal
%Method of moments
teta1_LNorm= log(mu) -0.5 * log(1 + std1^2/mu^2);
teta2_LNorm= sqrt(log( 1+std1^2/mu^2));
%Method of L_moments
teta2_LNorm_L=sqrt(2)*norminv((1+l2/l1)/2);
teta1_LNorm_L=log(l1)-(teta2_LNorm_L^2)/2;


%% Pearson Test

alpha= 0.10;
%Gumbel


k = round( 2 * (n^0.4) );

Ei= n * 1/k;

F=(0:1 / k:1);

limits= teta1_Gu - teta2_Gu * log( -log (F ) );
limits(1)=0;

oi=zeros(k,1);
for i=1:k
   
x1= limits (i);
x2 = limits(i+1);
oi (i) = sum (logical ( ord_Qm_Min>= x1 & ord_Qm_Min<x2 ));

end

disp('Observation and sample size:')
disp([sum(oi), n])


X2= sum( (oi - Ei).^2/ Ei);


%we want to define x2lim
%we use np=2 because it gumbel distribution

dof = (k - 2 -1);

X2lim=chi2inv( 1-alpha, dof);

if X2 > X2lim
    disp('Gumbel Test NOT passed, distribution is rejected')
else
    disp('Gumbel Test passed, distribution is accepted')
end
%         L_MOMENTS
limits1= teta1_Gu_L - teta2_Gu_L * log( -log (F ) );
limits1(1)=0;

oi1=zeros(k,1);
for i=1:k
   
x11= limits1 (i);
x21 = limits1(i+1);
oi1 (i) = sum (logical ( ord_Qm_Min>= x11 & ord_Qm_Min<x21 ));

end

disp('Observation and sample size:')
disp([sum(oi1), n])


X21= sum( (oi1 - Ei).^2/ Ei);


%we want to define x2lim
%we use np=2 because it gumbel distribution

dof = (k - 2 -1);

X2lim1=chi2inv( 1-alpha, dof);

if X21 > X2lim1
    disp('Gumbel Test NOT passed, distribution is rejected')
else
    disp('Gumbel Test passed, distribution is accepted')
end


%%

%Exponential

limits= -teta_ex*log(1-F);
limits(1)=0;

oi=zeros(k,1);
for i=1:k
   
x1= limits (i);
x2 = limits(i+1);
oi (i) = sum (logical ( ord_Qm_Min>= x1 & ord_Qm_Min<x2 ));

end

disp('Observation and sample size:')
disp([sum(oi), n])


X2= sum( (oi - Ei).^2/ Ei);


%we want to define x2lim
%we use np=2 because it gumbel distribution

dof = (k - 1 -1);

X2lim=chi2inv( 1-alpha, dof);


if X2 > X2lim
    disp('Exponential Test NOT passed, distribution is rejected')
else
    disp('Exponential Test passed, distribution is accepted')
end
%         L_MOMENTS
limits1= -teta_ex_L*log(1-F);
limits1(1)=0;

oi1=zeros(k,1);
for i=1:k
   
x11= limits1 (i);
x21 = limits1(i+1);
oi1 (i) = sum (logical ( ord_Qm_Min>= x11 & ord_Qm_Min<x21 ));

end

disp('Observation and sample size:')
disp([sum(oi1), n])


X21= sum( (oi1 - Ei).^2/ Ei);


%we want to define x2lim
%we use np=2 because it gumbel distribution

dof = (k - 1 -1);

X2lim1=chi2inv( 1-alpha, dof);

if X21 > X2lim1
    disp('Exponential Test NOT passed, distribution is rejected')
else
    disp('Exponential Test passed, distribution is accepted')
end

%%

%Lognormal

limits= exp(teta1_LNorm+teta2_LNorm*norminv(F));
limits(1)=0;

oi=zeros(k,1);
for i=1:k 
x1= limits (i);
x2 = limits(i+1);
oi (i) = sum (logical ( ord_Qm_Min>= x1 & ord_Qm_Min<x2 ));
end

disp('Observation and sample size:')
disp([sum(oi), n])
X2= sum( (oi - Ei).^2/ Ei);

dof = (k - 2 -1);

X2lim=chi2inv( 1-alpha, dof);


if X2 > X2lim
    disp('Lognormal Test NOT passed, distribution is rejected')
else
    disp('Lognormal Test passed, distribution is accepted')
end

%         L_MOMENTS
limits1=exp(teta1_LNorm_L+teta2_LNorm_L*norminv(F));
limits1(1)=0;

oi1=zeros(k,1);
for i=1:k
   
x11= limits1 (i);
x21 = limits1(i+1);
oi1 (i) = sum (logical ( ord_Qm_Min>= x11 & ord_Qm_Min<x21 ));

end

disp('Observation and sample size:');
disp([sum(oi1), n]);


X21= sum( (oi1 - Ei).^2/ Ei);


%we want to define x2lim
%we use np=2 because it gumbel distribution

dof = (k - 2 -1);

X2lim1=chi2inv( 1-alpha, dof);

if X21 > X2lim1
    disp('Lognormal Test NOT passed, distribution is rejected')
else
    disp('Lognormal Test passed, distribution is accepted')
end


% ---NORMAL



limits=teta1_Norm+teta2_Norm*norminv(F);
limits(1)=0;

oi=zeros(k,1);
for i=1:k 
x1= limits (i);
x2 = limits(i+1);
oi (i) = sum (logical ( ord_Qm_Min>= x1 & ord_Qm_Min<x2 ));
end

disp('Observation and sample size:')
disp([sum(oi), n])
X2= sum( (oi - Ei).^2/ Ei);

dof = (k - 2 -1);

X2lim=chi2inv( 1-alpha, dof);


if X2 > X2lim
    disp('NORMAL Test NOT passed, distribution is rejected')
else
    disp('NORMAL Test passed, distribution is accepted')
end

%         L_MOMENTS
limits1=teta1_Norm_L+teta2_Norm_L*norminv(F);
limits1(1)=0;

oi1=zeros(k,1);
for i=1:k
   
x11= limits1 (i);
x21 = limits1(i+1);
oi1 (i) = sum (logical ( ord_Qm_Min>= x11 & ord_Qm_Min<x21 ));

end

disp('Observation and sample size:')
disp([sum(oi1), n])


X21= sum( (oi1 - Ei).^2/ Ei);


%we want to define x2lim
%we use np=2 because it gumbel distribution

dof = (k - 2 -1);

X2lim1=chi2inv( 1-alpha, dof);

if X21 > X2lim1
    disp('NORMAL Test NOT passed, distribution is rejected')
else
    disp('NORMAL Test passed, distribution is accepted')
end


%%
%         ANDERSON_DARLING

P=normcdf((ord_Qm_Min-teta1_Norm_L)/teta2_Norm_L);
A=0;
for i=1:n
    A=A+((2*i-1)*log(P(i)))+(2*n+1-2*i)*log(1-P(i));
end
A2=-n-(1/n)*A;
ksi=0.167;
betta=0.229;
etta=1.147;
omega=0.0403+0.116*((A2-ksi)/betta)^(etta/0.861);
if omega > 0.347
    disp('Test NOT passed, distribution is rejected')
else
    disp('Test passed, distribution is accepted')
end

%%
T=1./(P);
figure(24)
semilogx(T,ord_Qm_Min,'*')

%%
F=1-(1/100);
F1=(1/100);
Q_m_RP100=teta1_Norm_L+teta2_Norm_L*norminv(F);
Q_m_RP100_1=teta1_Norm_L+teta2_Norm_L*norminv(F1);

%% 


%------------------PLATE -----------------------------

%*******************************************************
ord_Qp_Min = sort(Qp_Min, 'ascend');

n= length(yyp_cur);
k = 1:length(yyp_cur);
for i=1:length(yyp_cur)    
    Fs(i) = i/(length(yyp_cur)+1);
end

figure (25)
plot(ord_Qp_Min,Fs,'.-') %empirical distribution function
xlabel( 'sample values [m^3/s]')
ylabel( 'Weibull plotting postion') 


%for Gumbel distribution 
ui = -log( -log( Fs ) ); %Reduced variate

figure(26)
plot(ord_Qp_Min,ui, '*')
xlabel('sample values [m3/s]')
ylabel('reduced variate [-]')
title('Gumbel probability plot')% we cant say yet that are perfectly aligned INTERMEDIATE TYPE OF ALIGNMENT


%for exponential distribution

ui = -log(1 -Fs ); %Reduced variate
 

figure(27)
plot(ord_Qp_Min,ui, '*')
xlabel('sample values [m3/s]')
ylabel('reduced variate [-]')
title('Exponential probability plot') %We can say that it is a bit better than gumber but we need analytical expressions
%to justify the choice, we need statistical inference


%for normal distribution
mean_ord_Qp_Min=mean(ord_Qp_Min)
std_ord_Qp_Min=std(ord_Qp_Min)
ui= norminv (Fs,mean_ord_Qp_Min,std_ord_Qp_Min)



figure(28)
plot(ord_Qp_Min,ui, 'o', 'markersize', 12)

xlabel('sample values [m3/s]')
ylabel('reduced variate [-]')
title('Normal probability plot')
%set(gca,'fontsize', 20)


% lognormal distribution

y = log (ord_Qp_Min);
ui = norminv(Fs);


figure(29)
plot(y,ui, 'o', 'markersize', 12)

xlabel('log sample values [m3/s]')
ylabel('reduced variate [-]')
title(' Log Normal probability plot')


%% %-------------- Moments & L-Moments --------------
mu=mean(ord_Qp_Min);

%Gumbel
%------- MOMENTS -------
std1=std(ord_Qp_Min);
teta2_Gu=std1*sqrt(6)/pi;
gamma=0.5772;
teta1_Gu=mu-gamma*teta2_Gu;
%Method of L_moments
b0 = mu;
n=length(ord_Qp_Min);
w=((1:n)-1)/(n-1);

b1=sum(w'.*ord_Qp_Min)/n;  %moltiplication element by element w' transpose vector to be compliant with xi
w1=((1:n)-1).*((1:n)-2)/((n-1)*(n-2));
b2=sum( w1'.*ord_Qp_Min)/n;

l1=b0;
l2 = 2*b1 - b0;
l3 = 6*b2 - 6*b1 + b0;

teta2_Gu_L= l2/log(2);
teta1_Gu_L = l1-0.5772*teta2_Gu_L;

%exponential
teta_ex=mu;
%Method of L_moments
teta_ex_L=l1;

%Normal
teta1_Norm=mu;
teta2_Norm=std1;
%Method of L_moments
teta1_Norm_L=l1;
teta2_Norm_L=sqrt(pi)*l2;

% lognormal
%Method of moments
teta1_LNorm= log(mu) -0.5 * log(1 + std1^2/mu^2);
teta2_LNorm= sqrt(log( 1+std1^2/mu^2));
%Method of L_moments
teta2_LNorm_L=sqrt(2)*norminv((1+l2/l1)/2);
teta1_LNorm_L=log(l1)-(teta2_LNorm_L^2)/2;


%% Pearson Test

alpha= 0.10;
%Gumbel


k = round( 2 * (n^0.4) );

Ei= n * 1/k;

F=(0:1 / k:1);

limits= teta1_Gu - teta2_Gu * log( -log (F ) );
limits(1)=0;

oi=zeros(k,1);
for i=1:k
   
x1= limits (i);
x2 = limits(i+1);
oi (i) = sum (logical ( ord_Qp_Min>= x1 & ord_Qp_Min<x2 ));

end

disp('Observation and sample size:');
disp([sum(oi), n]);


X2= sum( (oi - Ei).^2/ Ei);


%we want to define x2lim
%we use np=2 because it gumbel distribution

dof = (k - 2 -1);

X2lim=chi2inv( 1-alpha, dof);

if X2 > X2lim
    disp('Gumbel Test NOT passed, distribution is rejected')
else
    disp('Gumbel Test passed, distribution is accepted')
end
%         L_MOMENTS
limits1= teta1_Gu_L - teta2_Gu_L * log( -log (F ) );
limits1(1)=0;

oi1=zeros(k,1);
for i=1:k
   
x11= limits1 (i);
x21 = limits1(i+1);
oi1 (i) = sum (logical ( ord_Qp_Min>= x11 & ord_Qp_Min<x21 ));

end

disp('Observation and sample size:');
disp([sum(oi1), n]);


X21= sum( (oi1 - Ei).^2/ Ei);


%we want to define x2lim
%we use np=2 because it gumbel distribution

dof = (k - 2 -1);

X2lim1=chi2inv( 1-alpha, dof);

if X21 > X2lim1
    disp('Gumbel Test NOT passed, distribution is rejected')
else
    disp('Gumbel Test passed, distribution is accepted')
end


%%

%Exponential

limits= -teta_ex*log(1-F);
limits(1)=0;

oi=zeros(k,1);
for i=1:k
   
x1= limits (i);
x2 = limits(i+1);
oi (i) = sum (logical ( ord_Qp_Min>= x1 & ord_Qp_Min<x2 ));

end

disp('Observation and sample size:');
disp([sum(oi), n]);


X2= sum( (oi - Ei).^2/ Ei);


%we want to define x2lim
%we use np=2 because it gumbel distribution

dof = (k - 1 -1);
X2lim=chi2inv( 1-alpha, dof);


if X2 > X2lim
    disp('Exponential Test NOT passed, distribution is rejected')
else
    disp('Exponential Test passed, distribution is accepted')
end
%         L_MOMENTS
limits1= -teta_ex_L*log(1-F);
limits1(1)=0;

oi1=zeros(k,1);
for i=1:k
   
x11= limits1 (i);
x21 = limits1(i+1);
oi1 (i) = sum (logical ( ord_Qp_Min>= x11 & ord_Qp_Min<x21 ));

end

disp('Observation and sample size:');
disp([sum(oi1), n]);


X21= sum( (oi1 - Ei).^2/ Ei);


%we want to define x2lim
%we use np=2 because it gumbel distribution

dof = (k - 1 -1);

X2lim1=chi2inv( 1-alpha, dof);

if X21 > X2lim1
    disp('Exponential Test NOT passed, distribution is rejected')
else
    disp('Exponential Test passed, distribution is accepted')
end

%%

%Lognormal

limits= exp(teta1_LNorm+teta2_LNorm*norminv(F));
limits(1)=0;

oi=zeros(k,1);
for i=1:k 
x1= limits (i);
x2 = limits(i+1);
oi (i) = sum (logical ( ord_Qp_Min>= x1 & ord_Qp_Min<x2 ));
end

disp('Observation and sample size:');
disp([sum(oi), n]);
X2= sum( (oi - Ei).^2/ Ei);

dof = (k - 2 -1);

X2lim=chi2inv( 1-alpha, dof);


if X2 > X2lim
    disp('Lognormal Test NOT passed, distribution is rejected')
else
    disp('Lognormal Test passed, distribution is accepted')
end

%         L_MOMENTS
limits1=exp(teta1_LNorm_L+teta2_LNorm_L*norminv(F));
limits1(1)=0;

oi1=zeros(k,1);
for i=1:k
   
x11= limits1 (i);
x21 = limits1(i+1);
oi1 (i) = sum (logical ( ord_Qp_Min>= x11 & ord_Qp_Min<x21 ));

end

disp('Observation and sample size:');
disp([sum(oi1), n]);


X21= sum( (oi1 - Ei).^2/ Ei);


%we want to define x2lim
%we use np=2 because it gumbel distribution

dof = (k - 2 -1);

X2lim1=chi2inv( 1-alpha, dof);

if X21 > X2lim1
    disp('Lognormal Test NOT passed, distribution is rejected')
else
    disp('Lognormal Test passed, distribution is accepted')
end


% ---NORMAL



limits=teta1_Norm+teta2_Norm*norminv(F);
limits(1)=0;

oi=zeros(k,1);
for i=1:k 
x1= limits (i);
x2 = limits(i+1);
oi (i) = sum (logical ( ord_Qp_Min>= x1 & ord_Qp_Min<x2 ));
end

disp('Observation and sample size:');
disp([sum(oi), n]);
X2= sum( (oi - Ei).^2/ Ei);

dof = (k - 2 -1);

X2lim=chi2inv( 1-alpha, dof);


if X2 > X2lim
    disp('NORMAL Test NOT passed, distribution is rejected')
else
    disp('NORMAL Test passed, distribution is accepted')
end

%         L_MOMENTS
limits1=teta1_Norm_L+teta2_Norm_L*norminv(F);
limits1(1)=0;

oi1=zeros(k,1);
for i=1:k
   
x11= limits1 (i);
x21 = limits1(i+1);
oi1 (i) = sum (logical ( ord_Qp_Min>= x11 & ord_Qp_Min<x21 ));

end

disp('Observation and sample size:');
disp([sum(oi1), n]);


X21= sum( (oi1 - Ei).^2/ Ei);


%we want to define x2lim
%we use np=2 because it gumbel distribution

dof = (k - 2 -1);

X2lim1=chi2inv( 1-alpha, dof);

if X21 > X2lim1
    disp('NORMAL Test NOT passed, distribution is rejected')
else
    disp('NORMAL Test passed, distribution is accepted')
end


%%
%         ANDERSON_DARLING
%      GUMBEL
P=exp(-exp(-(Qp_Min-teta1_Gu_L)/teta2_Gu_L));
A=0;
for i=1:n
    A=A+((2*i-1)*log(P(i)))+(2*n+1-2*i)*log(1-P(i));
end
A2=-n-(1/n)*A;
ksi=0.167;
betta=0.229;
etta=1.147;
omega=0.0403+0.116*((A2-ksi)/betta)^(etta/0.861);
if omega > 0.347
    disp('GUMBEL Anderson Test NOT passed, distribution is rejected')
else
    disp('GUMBEL Anderson Test passed, distribution is accepted')
end

%       LOGNORMAL
P=normcdf((log(ord_Qp_Min)-teta1_LNorm_L)/teta2_LNorm_L);
A=0;
for i=1:n
    A=A+((2*i-1)*log(P(i)))+(2*n+1-2*i)*log(1-P(i));
end
A2=-n-(1/n)*A;
ksi=0.167;
betta=0.229;
etta=1.147;
omega=0.0403+0.116*((A2-ksi)/betta)^(etta/0.861);
if omega > 0.347
    disp('LOGNORMAL Anderson Test NOT passed, distribution is rejected')
else
    disp('LOGNORMAL Anderson Test passed, distribution is accepted')
end
%%
T=1./(P);
figure(30)
semilogx(T,ord_Qp_Min,'*')

%%

F1=(1/100);
Q_p_RP100_1=exp(teta1_LNorm_L+teta2_LNorm_L*norminv(F1));