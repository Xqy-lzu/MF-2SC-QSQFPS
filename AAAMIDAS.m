function Output = AAAMIDAS(DataX,DataY,DataXdate,DataYdate,xlag,ylag,horizon,estStart,estEnd)


dispTime = 'full';

%%
MixedFreqData = MixFreqData(DataY,DataYdate,DataX,DataXdate,xlag,ylag,horizon,estStart,estEnd,dispTime);

EstX = MixedFreqData.EstX; 
EstXdate = MixedFreqData.EstXdate;
EstLagY = MixedFreqData.EstLagY;
EstY = MixedFreqData.EstY;

x.xmidas = EstX;
x.xmidasd = EstXdate;

OutX = MixedFreqData.OutX;
OutXdate = MixedFreqData.OutXdate;
OutLagY = MixedFreqData.OutLagY;
OutY= MixedFreqData.OutY;

xx.xmidas = OutX;
xx.xmidasd = OutXdate;

%poly的输入：'beta','betaNN','exp','LOG','sfun','almon'
%params的输入：beta 1~2个， betaNN 2~3个， exp 1~2个， LOG 1~2个
%beta与betaNN默认[1,5]; exp默认[-1 ,-0 ]'; LOG默认sfun默认3；3；xlag ; almon默认3
%comb的输入：'rt','es' {默认es}
poly='exp';
params=[-1 ,-0 ]';
comb='es';

TrainX=midas_X(x,poly,params,comb); 
 
TestX=midas_X(xx,poly,params,comb);

TrainY=EstY;

TestY=OutY;

TrainYL=EstLagY;

TestYL=OutLagY;

Output = struct('EstX',EstX,'EstXdate',EstXdate,'EstLagY',EstLagY,'EstY',EstY,'x',x,'xx',xx,'OutX',OutX,'OutXdate',OutXdate,'OutLagY',OutLagY,'OutY',OutY,'TrainX',TrainX,'TestX',TestX,'TrainY',TrainY,'TestY',TestY,'TrainYL',TrainYL,'TestYL',TestYL);
end

