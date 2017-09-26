% Reads in the training, validation, and test sets for group 11 project 1
% Performs a 5-fold least squaerror Regression on the data.
clear;
train(:,:,1) = csvread('C:\users\jeffrey\g11project01\LinearRegression\Data\train.00.csv');
train(:,:,2) = csvread('C:\users\jeffrey\g11project01\LinearRegression\Data\train.01.csv');
train(:,:,3) = csvread('C:\users\jeffrey\g11project01\LinearRegression\Data\train.02.csv');
train(:,:,4) = csvread('C:\users\jeffrey\g11project01\LinearRegression\Data\train.03.csv');
train(:,:,5) = csvread('C:\users\jeffrey\g11project01\LinearRegression\Data\train.04.csv');

valid(:,:,1) = csvread('C:\users\jeffrey\g11project01\LinearRegression\Data\validation.00.csv');
valid(:,:,2) = csvread('C:\users\jeffrey\g11project01\LinearRegression\Data\validation.01.csv');
valid(:,:,3) = csvread('C:\users\jeffrey\g11project01\LinearRegression\Data\validation.02.csv');
valid(:,:,4) = csvread('C:\users\jeffrey\g11project01\LinearRegression\Data\validation.03.csv');
valid(:,:,5) = csvread('C:\users\jeffrey\g11project01\LinearRegression\Data\validation.04.csv');

test = csvread('C:\users\jeffrey\g11project01\LinearRegression\Data\test.csv');

weights=zeros(5,3,2);
error=zeros(5,2);


%% Fitting and validation
for i=1:5

    % weights(:,:,1) is for Registered / Registered for the week
    weights (i,:,1) = polyfit(train(:,3,i),train(:,4,i),2);

    % weights(:,:,2) is for Registered / all for the week
    weights (i,:,2) = polyfit(train(:,3,i),train(:,5,i),2);

    
    % error(:,1) is for data normalized to regristered totals
    error(i,1) = sum( (myPoly( valid(:,3,i),weights(i,:,1) ) - valid(:,4,i)).^2 )/2;

    % error(:,2) is for data normalized to weekly totals
    error(i,2) = sum( (myPoly( valid(:,3,i),weights(i,:,2) ) - valid(:,5,i)).^2 )/2;

end


%% Plots
for i=1:5
    figure(i);
    hold off;
    plot(train(:,3,i), train(:,4,i),'.')
    hold on
    plot(0:6, myPoly(0:6,weights(i,:,1)),'r')
    title(cat(2,'Training Set for Fold ',num2str(i)));
    xlabel('Day of the week');
    ylabel('Fraction of registered riders (normalized to registered only)');
    set(gca,'XTickLabel',{'Sun','Mon','Tue','Wed','Thr','Fri','Sat'});
    
    figure(i+10);
    hold off;
    plot(train(:,3,i), train(:,4,i),'r.')
    hold on;
    plot(valid(:,3,i), valid(:,4,i),'.')
    plot(0:6, myPoly(0:6,[ -0.00433488 0.02524542 0.1113234 ]),'k')
   
%    plot(0:6, myPoly(0:6,weights(i,:,1)),'k')
    title(cat(2,'Training and Validation Sets for Fold ',num2str(i)));
    xlabel('Day of the week');
    ylabel('Fraction of registered riders (normalized to all riders)');
    set(gca,'XTickLabel',{'Sun','Mon','Tue','Wed','Thr','Fri','Sat'});

end

best(1,1)=find(error(:,1)==min(error(:,1)));
best(1,2)=find(error(:,2)==min(error(:,2)));

% FinalError(1,1) is for data normalized to regristered totals
FinalError(1,1) = sum( (myPoly( test(:,3),weights(best(1),:,1) ) - test(:,4) ).^2 )/2;
% FinalError(1,2) is for data normalized to regristered totals
FinalError(1,2) = sum( (myPoly( test(:,3),weights(best(2),:,2) ) - test(:,5) ).^2 )/2;


figure(21);
hold off;
plot(test(:,3), test(:,4),'.')
hold on
plot(0:6, myPoly(0:6,weights(best(1),:,1)),'r')
title('Test Set');
xlabel('Day of the week');
ylabel('Fraction of registered riders (normalized to registered only)');
set(gca,'XTickLabel',{'Sun','Mon','Tue','Wed','Thr','Fri','Sat'});

figure(22);
hold off;
plot(test(:,3), test(:,5),'.')
hold on
plot(0:6, myPoly(0:6,weights(best(2),:,2)),'r')
title('Test Set Maximum Likelihood');
xlabel('Day of the week');
ylabel('Fraction of registered riders (normalized to all riders)');
set(gca,'XTickLabel',{'Sun','Mon','Tue','Wed','Thr','Fri','Sat'});

figure(23);
hold off;
plot(test(:,3), test(:,5),'.')
hold on
plot(0:6, myPoly(0:6,[ -0.00433488 0.02524542 0.1113234 ]),'r')
title('Test Set Gradient Descent');
xlabel('Day of the week');
ylabel('Fraction of registered riders (normalized to all riders)');
set(gca,'XTickLabel',{'Sun','Mon','Tue','Wed','Thr','Fri','Sat'});

