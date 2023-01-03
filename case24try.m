%% uncomment this when not running through the EULER cluster
addpath( ...
    '/cluster/home/hlouisa/toolbox/matpower7.1/lib', ...
    '/cluster/home/hlouisa/toolbox/matpower7.1/lib/t', ...
    '/cluster/home/hlouisa/toolbox/matpower7.1/data', ...
    '/cluster/home/hlouisa/toolbox/matpower7.1/most/lib', ...
    '/cluster/home/hlouisa/toolbox/matpower7.1/most/lib/t', ...
    '/cluster/home/hlouisa/toolbox/matpower7.1/mp-opt-model/lib', ...
    '/cluster/home/hlouisa/toolbox/matpower7.1/mp-opt-model/lib/t', ...
    '/cluster/home/hlouisa/toolbox/matpower7.1/mips/lib', ...
    '/cluster/home/hlouisa/toolbox/matpower7.1/mips/lib/t', ...
    '/cluster/home/hlouisa/toolbox/matpower7.1/mptest/lib', ...
    '/cluster/home/hlouisa/toolbox/matpower7.1/mptest/lib/t', ...
    '/cluster/home/hlouisa/toolbox/matpower7.1/extras/maxloadlim', ...
    '/cluster/home/hlouisa/toolbox/matpower7.1/extras/maxloadlim/tests', ...
    '/cluster/home/hlouisa/toolbox/matpower7.1/extras/maxloadlim/examples', ...
    '/cluster/home/hlouisa/toolbox/matpower7.1/extras/misc', ...
    '/cluster/home/hlouisa/toolbox/matpower7.1/extras/reduction', ...
    '/cluster/home/hlouisa/toolbox/matpower7.1/extras/sdp_pf', ...
    '/cluster/home/hlouisa/toolbox/matpower7.1/extras/se', ...
    '/cluster/home/hlouisa/toolbox/matpower7.1/extras/smartmarket', ...
    '/cluster/home/hlouisa/toolbox/matpower7.1/extras/state_estimator', ...
    '/cluster/home/hlouisa/toolbox/matpower7.1/extras/syngrid/lib', ...
    '/cluster/home/hlouisa/toolbox/matpower7.1/extras/syngrid/lib/t', ...
    '-end' );
%%%%
% uncomment this when not running throught the EULER cluster
load('hourlyDemandBus.mat') 
%%%
angle=[];
magnitude=[];
TotalDC=[];
tic
%define_constants;
mpctry = loadcase('System.m');
for j=1:5000
    for i=1:24
        mpctry.bus(i, 3)= hourlyDemandBus(i,j);
    end
    resu=mpctry.bus(:,3);
    case24new=runpf(mpctry);
    result_angle=case24new.bus(:,9);
    result_mag=case24new.bus(:,8);
    G_index=case24new.gen(:,1);
    GenActivePower=case24new.gen(:,2);
    result_GenActivePower=[accumarray(G_index,GenActivePower);0];
    GenReactivePower=case24new.gen(:,3);
    result_GenReactivePower=[accumarray(G_index,GenReactivePower);0]
    LoadActivePower=case24new.bus(:,3);
    LoadReactivePower=case24new.bus(:,4);
    NetREALPower=[abs(result_GenActivePower-LoadActivePower)];
    NetReactivePower=[abs(result_GenReactivePower-LoadReactivePower)];
    edgeindex=[case24new.branch(:,1),case24new.branch(:,2)];
    fill=[];
    for k=1:24
        matrix=[NetREALPower(k),NetReactivePower(k),result_mag(k),result_angle(k)];
        fill=[fill,matrix];
    end
    TotalDC=[TotalDC;fill];
end
%%%%%%%%%%%%add label 1:PQ, 2:PV, 3: ref
Bustype=case24new.bus(:,2);
totbustype=[];
for j=1:24
    matrixbus=[Bustype(j),Bustype(j),Bustype(j),Bustype(j)];
    totbustype=[totbustype,matrixbus];
end
%%%%%%%%%%%% add legend P,Q,V,A
full={};
labelP=sprintfc(['P%d'], 1:24);
listP=[labelP];
labelQ=sprintfc(['Q%d'], 1:24);
listQ=[labelQ];
labelV=sprintfc(['V%d'], 1:24);
listV=[labelV];
labelA=sprintfc(['A%d'], 1:24);
listA=[labelA];
for i=1:24
    list=[listP(i),listQ(i),listV(i),listA(i)];
    full=[full,list];
end 
%%%%%%%%%%%%%%
full;
strfull=string(full); %cell array needs to be converted in string array so that it can be concatenated
Total=[strfull;totbustype;TotalDC];
toc 
t=toc
%writematrix(edgeindex, '/Users/louisahillegaart/Documents/ETH MSC/MAVT/SemesterThesis/Matlab/Case24/BIGEdgeIndex.xlsx')
%writematrix(Total, '/Users/louisahillegaart/Documents/ETH MSC/MAVT/SemesterThesis/Matlab/Case24/newcasenew24AC_TRYnonlabeled.xlsx')

%end