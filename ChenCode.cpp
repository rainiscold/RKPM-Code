//
//  main.cpp
//  HW4_SE279_WI15
//
//  Created by Ivana Escobar on 2/13/15.
//
//

#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <math.h>
#include <cmath>
#include <iomanip>
#include </Volumes/IVANA_S/UCSD/SE279/eigen-eigen-10219c95fe65/Eigen/LU>

# define PI 3.141592653589793238462643383279502884L /* pi */

using namespace Eigen;
using namespace std;

//------------------------------------------------------------------------------------------------
//  GAUSS QUADRATURE INTEGRATION
//------------------------------------------------------------------------------------------------

//--------------- GAUSS WEIGHTS ---------------
RowVectorXd gaussWeights (int gaussOrder){
    
    RowVectorXd quadrature_weights(gaussOrder);
    
    if (gaussOrder == 2){
        quadrature_weights << 1.000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000, 1.000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000;
    }
    else if (gaussOrder == 3){
        quadrature_weights <<0.8888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888889, 0.5555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555556, 0.5555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555556;
    }
    else if (gaussOrder == 4){
        quadrature_weights << 0.6521451548625461426269360507780005927646513041661064595074706804812481325340896482780162322677418404902018960952364978455755577496740182191429757016783303751407135229556360801973666260481564013273531860737119707353160256000107787211587578617532049337456560923057986412084590467808124974086, 0.6521451548625461426269360507780005927646513041661064595074706804812481325340896482780162322677418404902018960952364978455755577496740182191429757016783303751407135229556360801973666260481564013273531860737119707353160256000107787211587578617532049337456560923057986412084590467808124974086, 0.3478548451374538573730639492219994072353486958338935404925293195187518674659103517219837677322581595097981039047635021544244422503259817808570242983216696248592864770443639198026333739518435986726468139262880292646839743999892212788412421382467950662543439076942013587915409532191875025701, 0.3478548451374538573730639492219994072353486958338935404925293195187518674659103517219837677322581595097981039047635021544244422503259817808570242983216696248592864770443639198026333739518435986726468139262880292646839743999892212788412421382467950662543439076942013587915409532191875025701;
    }
    else if (gaussOrder == 8){
        quadrature_weights <<0.3626837833783619829651504492771956121941460398943305405248230675666867347239066773243660420848285095502587699262967065529258215569895173844995576007862076842778350382862546305771007553373269714714894268328780431822779077846722965535548199601402487767505928976560993309027632737537826127502150514997832, 0.3626837833783619829651504492771956121941460398943305405248230675666867347239066773243660420848285095502587699262967065529258215569895173844995576007862076842778350382862546305771007553373269714714894268328780431822779077846722965535548199601402487767505928976560993309027632737537826127502150514997832, 0.3137066458778872873379622019866013132603289990027349376902639450749562719421734969616980762339285560494275746410778086162472468322655616056890624276469758994622503118776562559463287222021520431626467794721603822601295276898652509723185157998353156062419751736972560423953923732838789657919150514997832, 0.3137066458778872873379622019866013132603289990027349376902639450749562719421734969616980762339285560494275746410778086162472468322655616056890624276469758994622503118776562559463287222021520431626467794721603822601295276898652509723185157998353156062419751736972560423953923732838789657919150514997832, 0.2223810344533744705443559944262408844301308700512495647259092892936168145704490408536531423771979278421592661012122181231114375798525722419381826674532090577908613289536840402789398648876004385697202157482063253247195590228631570651319965589733545440605952819880671616779621183704306688233150514997832, 0.2223810344533744705443559944262408844301308700512495647259092892936168145704490408536531423771979278421592661012122181231114375798525722419381826674532090577908613289536840402789398648876004385697202157482063253247195590228631570651319965589733545440605952819880671616779621183704306688233150514997832, 0.1012285362903762591525313543099621901153940910516849570590036980647401787634707848602827393040450065581543893314132667077154940308923487678731973041136073584690533208824050731976306575729205467961435779467552492328730055025992954089946676810510810729468366466585774650346143712142008566866150514997832, 0.1012285362903762591525313543099621901153940910516849570590036980647401787634707848602827393040450065581543893314132667077154940308923487678731973041136073584690533208824050731976306575729205467961435779467552492328730055025992954089946676810510810729468366466585774650346143712142008566866150514997832;
    }
    
    return quadrature_weights;
}

//--------------- GAUSS ABSCISSA ---------------
RowVectorXd gaussAbcissa (int gaussOrder){
    
    RowVectorXd quadrature_abcissa(gaussOrder);
    
    if (gaussOrder == 2){
        quadrature_abcissa << -0.5773502691896257645091487805019574556476017512701268760186023264839776723029333456937153955857495252252087138051355676766566483649996508262705518373647912161760310773007685273559916067003615583077550051041144223011076288835574182229739459904090157105534559538626730166621791266197964892168, 0.5773502691896257645091487805019574556476017512701268760186023264839776723029333456937153955857495252252087138051355676766566483649996508262705518373647912161760310773007685273559916067003615583077550051041144223011076288835574182229739459904090157105534559538626730166621791266197964892168;
    }
    else if (gaussOrder == 3){
        quadrature_abcissa << 0.0, -0.774596669241483377035853079956479922166584341058318165317514753222696618387395806703857475371734703583260441372189929402637908087832729923135978349224240702213750958202698716256783906245777858513169283405612501838634682531972963691092925710263188052523534528101729260090115562126394576188, 0.774596669241483377035853079956479922166584341058318165317514753222696618387395806703857475371734703583260441372189929402637908087832729923135978349224240702213750958202698716256783906245777858513169283405612501838634682531972963691092925710263188052523534528101729260090115562126394576188;
    }
    else if (gaussOrder == 4){
        quadrature_abcissa << -0.3399810435848562648026657591032446872005758697709143525929539768210200304632370344778752804355548115489602395207464932135845003241712491992776363684338328221538611182352836311104158340621521124125023821932864240034767086752629560943410821534146791671405442668508151756169732898924953195536, 0.3399810435848562648026657591032446872005758697709143525929539768210200304632370344778752804355548115489602395207464932135845003241712491992776363684338328221538611182352836311104158340621521124125023821932864240034767086752629560943410821534146791671405442668508151756169732898924953195536, -0.8611363115940525752239464888928095050957253796297176376157219209065294714950488657041623398844793052105769209319781763249637438391157919764084938458618855762872931327441369944290122598469710261906458681564745219362114916066097678053187180580268539141223471780870198639372247416951073770551, 0.8611363115940525752239464888928095050957253796297176376157219209065294714950488657041623398844793052105769209319781763249637438391157919764084938458618855762872931327441369944290122598469710261906458681564745219362114916066097678053187180580268539141223471780870198639372247416951073770551;
    }
    else if (gaussOrder == 8){
        quadrature_abcissa << -0.1834346424956498049394761423601839806667578129129737823171884736992044742215421141160682237111233537452676587642867666089196012523876865683788569995160663568104475551617138501966385810764205532370882654749492812314961247764619363562770645716456613159405134052985058171969174306064445289638150514997832, 0.1834346424956498049394761423601839806667578129129737823171884736992044742215421141160682237111233537452676587642867666089196012523876865683788569995160663568104475551617138501966385810764205532370882654749492812314961247764619363562770645716456613159405134052985058171969174306064445289638150514997832, -0.5255324099163289858177390491892463490419642431203928577508570992724548207685612725239614001936319820619096829248252608507108793766638779939805395303668253631119018273032402360060717470006127901479587576756241288895336619643528330825624263470540184224603688817537938539658502113876953598879150514997832, 0.5255324099163289858177390491892463490419642431203928577508570992724548207685612725239614001936319820619096829248252608507108793766638779939805395303668253631119018273032402360060717470006127901479587576756241288895336619643528330825624263470540184224603688817537938539658502113876953598879150514997832, -0.7966664774136267395915539364758304368371717316159648320701702950392173056764730921471519272957259390191974534530973092653656494917010859602772562074621689676153935016290342325645582634205301545856060095727342603557415761265140428851957341933710803722783136113628137267630651413319993338002150514997832, 0.7966664774136267395915539364758304368371717316159648320701702950392173056764730921471519272957259390191974534530973092653656494917010859602772562074621689676153935016290342325645582634205301545856060095727342603557415761265140428851957341933710803722783136113628137267630651413319993338002150514997832, -0.960289856497536231683560868569472990428235234301452038271639777372424897743419284439438959263312268310424392817294176210238958155217128547937364220490969970043398261832663734680878126355334692786735966348087059754254760392931853386656813286884261347489628923208763998895240977248938732425615051499783203, 0.960289856497536231683560868569472990428235234301452038271639777372424897743419284439438959263312268310424392817294176210238958155217128547937364220490969970043398261832663734680878126355334692786735966348087059754254760392931853386656813286884261347489628923208763998895240977248938732425615051499783203;
    }
    
    return quadrature_abcissa;
}

//--------------- GAUSS POINTS AND WEIGHTS ---------------
tuple<RowVectorXd, RowVectorXd> gaussPtsandWeights(int gaussOrder, RowVectorXd x_i, double domain, double h){
    // Finds the corresponding weights and abscissa in the grid of a 1D domain
    
    RowVectorXd gpts_original;
    RowVectorXd wpts_original;
    gpts_original = gaussAbcissa(gaussOrder);
    wpts_original = gaussWeights(gaussOrder);
    
    MatrixXd x_integration(gpts_original.size(), x_i.size()-1);
    MatrixXd weights(wpts_original.size(), x_i.size()-1);
    RowVectorXd gpts(gpts_original.size());
    RowVectorXd wpts(wpts_original.size());
    
    double a;
    double b;
    
    int k;
    
    RowVectorXd x_g(gpts_original.size() * (x_i.size()-1));
    RowVectorXd w_g(wpts_original.size() * (x_i.size()-1));
    tuple<RowVectorXd, RowVectorXd> gPtsandWeights;
    
    for (int i = 0; i < (x_i.size()-1); i++){
        a = domain/(h-1) * (i);
        b = domain/(h-1) * (i+1);
        // cout << "a " << a << "......b "  << b << endl;
        for (int j = 0; j < gpts.size(); j++){
            gpts(j) = ((b-a)/2) * gpts_original(j) + ((b+a)/2);
            x_integration(j,i) = x_integration(j,i) + gpts(j);
        }
        for (int k = 0; k < wpts.size(); k++){
            wpts(k) = ((b-a)/2) * wpts_original(k);
            weights(k,i) = weights(k,i) + wpts(k);
        }
    }
    
    // Takes the weights and integration point matrices and makes them row vectors
    k = 0;
    for (int i = 0; i < (x_i.size()-1); i++){
        for (int j = 0; j < gpts.size(); j++){
            x_g(k) = x_integration(j,i);
            k++;
        }
    }
    k = 0;
    for (int i = 0; i < (x_i.size()-1); i++){
        for (int j = 0; j < wpts.size(); j++){
            w_g(k) = weights(j,i);
            k++;
        }
    }

    gPtsandWeights = make_tuple(x_g,w_g);
    return gPtsandWeights;
}

//------------------------------------------------------------------------------------------------
//  BASIS AND KERNEL FUNCTIONS
//  (Using Gauss Quadrature)
//------------------------------------------------------------------------------------------------

//  Instead of using the former evaluation points, these functions will use the new Gauss points
//  created by the gauss integrationg scheme above.
//  (i.e. x changes to x_g)

//--------------- RK BASIS FUNCTION ---------------
VectorXd RKbasisFunction(int basisOrder, double x_i, double x_g){
    
    VectorXd p_RK(basisOrder+1);
    
    for (int i = 0; i <= basisOrder; i++)
    {
        p_RK(i) = pow((x_g - x_i), i);
    }
    return p_RK;
}

//--------------- RK DERIVATIVE BASIS FUNCTION ---------------
VectorXd RKderivBasisFunction(int basisOrder, double x_i, double x_g){
    
    VectorXd pDerivRK(basisOrder+1);
    //  imposing the first value to be 0.
    pDerivRK(0) = 0.;
    
    for (int i = 1; i <= basisOrder; i++)
    {
        pDerivRK(i) = i * pow((x_g - x_i), (i - 1));
    }
    return pDerivRK;
}

//--------------- Cubic B-Spline KERNEL FUNCTION --------------
//  C2 Continuity
double kernelFxn (double a, double x_i, double x_g){
    
    double phi;
    
    double z_i = abs((x_g - x_i)) / a;
    if (z_i >= 0 && z_i <= .5){
        phi = ((2./3.) - (4 * pow(z_i, 2)) + (4 * pow(z_i,3)));
    }
    else if (z_i > .5 && z_i <= 1){
        phi = ((4./3.) - (4 * z_i) + (4 * pow(z_i, 2)) - ((4./3.) * pow(z_i,3)));
    }
    else{
        phi = 0;
    }
    return phi;
}

//--------------- DERIVATIVE Cubic B-Spline KERNEL FUNCTION --------------
double derivKernelFxn (double a, double x_i, double x_g){
    
    double derivPhi;
    
    double z_i = abs((x_g - x_i)) / a;
    
    if (z_i >= 0. && z_i <= .5){
        derivPhi = ((-8. * z_i) + (12. * pow(z_i, 2.))) / a;
    }
    else if (z_i > .5 && z_i <= 1.){
        derivPhi = (-4. + (8. * z_i) - (4. * pow(z_i, 2.))) / a;
    }
    else{
        derivPhi = 0.;
    }
    
    if ((x_g - x_i) < 0.){
        
        derivPhi = derivPhi * -1;
    }
    
    return derivPhi;
}

//--------------- Box/Hat KERNEL FUNCTION --------------
//  C(-1) Continuity
double BoxkernelFxn (double a, double x_i, double x_g){
    
    double phi;
    
    double z_i = abs((x_g - x_i)) / a;
    if (z_i >= 0 && z_i <= 1){
        phi = 1/(2*a);
    }
    else{
        phi = 0;
    }
    return phi;
}

//--------------- DERIVATIVE Box/Hat KERNEL FUNCTION --------------
double derivBoxkernelFxn (double a, double x_i, double x_g){
    
    double derivPhi;
    
    double z_i = abs((x_g - x_i)) / a;
    if (z_i >= 0 && z_i <= 1){
        derivPhi = 0;
    }
    else{
        derivPhi = 0;
    }
    return derivPhi;
}

//--------------- Tent B-Spline KERNEL FUNCTION --------------
//  C0 Continuity
double TentkernelFxn (double a, double x_i, double x_g){
    
    double phi;
    
    double z_i = abs((x_g - x_i)) / a;
    if (z_i >= 0. && z_i <= 1.){
        phi = (-1. * z_i) + 1;
    }
    else{
        phi = 0;
    }
    return phi;
}

//--------------- Tent B-Spline KERNEL FUNCTION --------------
double DerivTentkernelFxn (double a, double x_i, double x_g){
    
    double derivPhi;
    
    double z_i = abs((x_g - x_i)) / a;
    if (z_i >= 0. && z_i <= 1.){
        derivPhi = -1.;
    }
    else{
        derivPhi = 0;
    }
    return derivPhi;
}


//------------------------------------------------------------------------------------------------
//  SINUSOIDAL FUNCTION
//  (Using Gauss Integration)
//------------------------------------------------------------------------------------------------

//  Exact function evaluated at the integration points, x_i, (to be used in creating the
//  approximated function) and the evaluation points, x_g, (to be used in error analysis)

 tuple<RowVectorXd, RowVectorXd, RowVectorXd> uFuctions(RowVectorXd x_i, int gaussOrder, double domain, double h, int k){
     
     tuple<RowVectorXd, RowVectorXd> gPtsandWeights = gaussPtsandWeights(gaussOrder, x_i, domain, h);
     RowVectorXd x_g;
     x_g = get<0>(gPtsandWeights);
     RowVectorXd w_g;
     w_g = get<1>(gPtsandWeights);
     
     RowVectorXd u_i(x_i.size());
     RowVectorXd u_ig(x_g.size());
     RowVectorXd du_ig(x_g.size());
     
     tuple<RowVectorXd, RowVectorXd, RowVectorXd> u_iAndu_ig;
     

     for (int i = 0; i < x_i.size(); i++){
         u_i(i) = sin(k * x_i(i));
     }
     for (int i = 0; i < x_g.size(); i++){
         u_ig(i) = sin(k * x_g(i));
         du_ig(i) = k * cos(k * x_g(i));
     }
     
     u_iAndu_ig = make_tuple(u_i, u_ig, du_ig);
     return u_iAndu_ig;
 }

//------------------------------------------------------------------------------------------------
//  REPRODUCING KERNEL APPROXIMATION METHOD
//  (with Gauss Integration)
//------------------------------------------------------------------------------------------------

//--------------- RK COEFFICIENTS ---------------
VectorXd RKCoefficients(int basisOrder, VectorXd u_i, VectorXd x_i, double x_g, double a){
    
    MatrixXd M(basisOrder+1, basisOrder+1);
    VectorXd sum(basisOrder+1);
    VectorXd b(basisOrder+1);
    
    VectorXd p_RK(basisOrder+1);
    double phi;
    
    M = 0. * M;
    sum = 0. * sum;
    for (int i = 0; i < x_i.size(); i++){
        p_RK = RKbasisFunction(basisOrder, x_i(i), x_g);
        phi = kernelFxn(a, x_i(i), x_g);
        
        M = M + (p_RK * p_RK.transpose()) * phi;
        
        sum = sum + (p_RK * phi * u_i(i));
    }
    b = M.inverse()*sum;
    return b;
}

//--------------- 1-D RK APPROXIMATION ---------------
RowVectorXd RKApproximation(int basisOrder, RowVectorXd x_i, RowVectorXd u_i, double a, VectorXd x_g){
    
    RowVectorXd uh_RK(x_g.size());
    VectorXd b_RK;
    
    for(int i = 0; i < x_g.size(); i++){
        b_RK = RKCoefficients(basisOrder, u_i, x_i, x_g(i), a);
        uh_RK(i) = b_RK(0);
    }
    cout << "The RK uh_g are:\n [" << uh_RK << "]" << endl;
    cout << endl;
    
    return uh_RK;
}

//--------------- DERIVATIVE RK COEFFICIENTS ---------------
VectorXd derivativeRKCoefficients(int basisOrder, RowVectorXd u_i, RowVectorXd x_i, double x_g, double a){
    
    MatrixXd M(basisOrder+1, basisOrder+1);
    MatrixXd Minv(basisOrder+1, basisOrder+1);
    MatrixXd derivM(basisOrder+1, basisOrder+1);
    
    VectorXd p_RK(basisOrder+1);
    VectorXd p_RK2(basisOrder+1);
    VectorXd pDerivRK;
    VectorXd pDerivRK2;
    
    double phi;
    double phi2;
    double derivPhi;
    double derivPhi2;

    VectorXd dsum(basisOrder+1);
    VectorXd db(basisOrder+1);
    
    M = 0. * M;
    derivM = 0. * derivM;
    dsum = 0. * dsum;
    Minv = 0. * Minv;
    
    for (int i = 0; i < x_i.size(); i++){
        p_RK = RKbasisFunction(basisOrder, x_i(i), x_g);
        phi = kernelFxn(a, x_i(i), x_g);
            
        pDerivRK = RKderivBasisFunction(basisOrder, x_i(i), x_g);
        derivPhi = derivKernelFxn (a, x_i(i), x_g);
            
        M = M + (p_RK * p_RK.transpose()) * phi;
            
        derivM = derivM + ((pDerivRK * p_RK.transpose() * phi) +
                           (p_RK * pDerivRK.transpose() * phi) +
                           (p_RK * p_RK.transpose() * derivPhi));
    }
    Minv = M.inverse();
    //cout << " Moment " << endl << M << endl <<
    //"derivM" << endl << derivM << endl << endl;
    for (int i = 0; i < x_i.size(); i++){
        p_RK2 = RKbasisFunction(basisOrder, x_i(i), x_g);
        phi2 = kernelFxn(a, x_i(i), x_g);
        
        pDerivRK2 = RKderivBasisFunction(basisOrder, x_i(i), x_g);
        derivPhi2 = derivKernelFxn (a, x_i(i), x_g);
        
        
        dsum = dsum + ((-1 * derivM * Minv * p_RK2 * phi2) +
                       (pDerivRK2 * phi2) +
                       (p_RK2 * derivPhi2)) * u_i(i);
        
       // cout << " iteration " << i << endl <<
        //"dsum" << endl << dsum << endl << endl;
    }
    db = Minv * dsum;
    return db;
}

//--------------- 1-D DERIVATIVE RK APPROXIMATION ---------------
RowVectorXd derivativeRKApproximation(int basisOrder, RowVectorXd x_i, RowVectorXd u_i, double a, RowVectorXd x_g){
    
    RowVectorXd duh_RK(x_g.size());
    VectorXd db_RK;
    
    for(int i = 0; i < x_g.size(); i++){
        db_RK = derivativeRKCoefficients(basisOrder, u_i, x_i, x_g(i), a);
        duh_RK(i) = db_RK(0);
    }
    cout << "The derivative RK duh_g are:\n [" << duh_RK << "]" << endl;
    cout << endl;
    return duh_RK;
}

//------------------------------------------------------------------------------------------------
//  ERROR ANALYSIS
//------------------------------------------------------------------------------------------------
//--------------- L2 and H1 Norm ---------------
tuple<double, double> L2andH1norms(RowVectorXd x_i, RowVectorXd x_g, RowVectorXd u_ig, RowVectorXd uh_RK, RowVectorXd du_ig, RowVectorXd duh_RK, int gaussOrder, double domain, double h){
    
    tuple<RowVectorXd, RowVectorXd> gPtsandWeights = gaussPtsandWeights(gaussOrder, x_i, domain, h);
    RowVectorXd w_g;
    w_g = get<1>(gPtsandWeights);
    
    tuple<double, double> L2andH1;

    double errorSum = 0.;
    double derivErrorSum = 0.;
    double L2norm = 0.;
    double H1norm = 0.;

    for (int i = 0; i < x_g.size(); i++){
        errorSum = errorSum + (pow((u_ig(i) - uh_RK(i)), 2) * w_g(i));
        derivErrorSum = derivErrorSum + (pow((du_ig(i) - duh_RK(i)), 2) * w_g(i));
    }

    L2norm = sqrt(errorSum);
    H1norm = sqrt(derivErrorSum);

    cout << "L2 norm: " << L2norm << endl << "H1 seminorm: " << H1norm << endl;
    cout << endl;
    
    L2andH1 = make_tuple(L2norm, H1norm);
    return L2andH1;
}

//------------------------------------------------------------------------------------------------
//  END: FUNCTIONS
//------------------------------------------------------------------------------------------------

/*  
                RK Approximation of a Sinusoidal Function
 */

int main(int argc, const char * argv[])
{
    //--------------------------------------------------------------------
    //  INPUTS
    //--------------------------------------------------------------------
    cout << "INPUTS: " << endl;
    //--------------- Domain ---------------
    double domain = 2 * PI;
    
    //--------------- Basis Order --------------
    //  Use linear and quadratic bases
    int basisOrder = 2;
    
    //--------------- Gauss Order --------------
    //  Use 8 integration points per grid space
    int gaussOrder = 8;
    
    //--------------- f(x) to be approximated --------------
    int k = 2;
    
    //--------------- Domain Intervals ---------------
    //RowVectorXd h(1);
    //h << 10, 20, 40, 80, 160, 320;
    double h = 10;
    
    //--------------- Initializing L2 and H1 norms ---------------
    double L2norm;
    double H1norm;
        
    //-------------- Support size ---------------
    double aconstant = 2.0;
    double a = aconstant * (domain/(h-1));
    
    //--------------- Nodes ---------------
    RowVectorXd x_i(h);
    for (int i = 0; i < h; i++){
        x_i(i) =  i * (domain/ (h - 1));
    }
    cout << "Here are the node values x_i:\n" << x_i << endl;
    cout << endl;
    
    //--------------- Gauss Evaluation Points ---------------
    tuple<RowVectorXd, RowVectorXd> gPtsandWeights = gaussPtsandWeights(gaussOrder, x_i, domain, h);
    RowVectorXd x_g;
    x_g = get<0>(gPtsandWeights);
    
    RowVectorXd x_gSquared(x_g.size());
    for (int i = 0; i < x_g.size(); i++){
        x_gSquared(i) = pow(x_g(i), 2);
    }
    cout << "Here are the values of the evaluation points x_g:\n" << x_g << endl;
    cout << endl;
        
    //
    // Value of the exact function at the nodes
    tuple<RowVectorXd, RowVectorXd, RowVectorXd> u_iAndu_ig = uFuctions(x_i, gaussOrder, domain, h, k);
    RowVectorXd u_i;
    u_i = get<0>(u_iAndu_ig);
    RowVectorXd u_ig;
    u_ig = get<1>(u_iAndu_ig);
    RowVectorXd du_ig;
    du_ig = get<2>(u_iAndu_ig);
    
    cout << "Here are the exact function values at the nodes u_i:\n" << endl << u_i << endl;
    cout << endl;
    cout << "Here are the exact function values at the evaluation points u_ig:\n" << endl << u_ig << endl;
    cout << endl;
    cout << "Here are the derivatives of the exact function values at the evaluation points du_ig:\n" << endl << du_ig << endl;
    cout << endl;
    
    
    //--------------------------------------------------------------------
    //  INITIALIZING RK FUNCTIONS
    //--------------------------------------------------------------------
    cout << endl;
    cout << "RESULTS: " << endl;

    RowVectorXd uh_RK = RKApproximation(basisOrder, x_i, u_i, a, x_g);
    RowVectorXd duh_RK = derivativeRKApproximation(basisOrder, x_i, u_i, a, x_g);
    
    //--------------------------------------------------------------------
    //  ERROR ANALYSIS
    //--------------------------------------------------------------------
    cout << endl;
    cout << "ERROR ANALYSIS: " << endl << endl;
    
    //--------------- Difference Vectors ---------------
    RowVectorXd L2diff(x_g.size());
    RowVectorXd H1diff(x_g.size());
    
    L2diff = u_ig - uh_RK;
    H1diff = du_ig - duh_RK;
    
    cout << "L2 diff: " << endl << L2diff << endl << endl << "H1 diff: " << endl << H1diff << endl;
    cout << endl;
    
    tuple<double, double> L2andH1 = L2andH1norms(x_i, x_g, u_ig, uh_RK, du_ig, duh_RK, gaussOrder, domain, h);
    L2norm = get<0>(L2andH1);
    H1norm = get<1>(L2andH1);
        
    cout << "L2 norm " << endl << setprecision(15) << L2norm << endl << "H1 norm " << endl << setprecision(15) << H1norm << endl;
    
    //--------------------------------------------------------------------
    //  WRITE TO .TXT FILE
    //--------------------------------------------------------------------
    
    //--------------- Open Writer ---------------
    
    ofstream writer("/Volumes/IVANA_S/UCSD/SE279/HW4/txtFiles/Homework4SE279.txt"); // name of file
    
    //  Verify if the string output can be written onto a file
    if (! writer){
        cout << "Error opening file" << endl;
        return -1;
    } else {
        writer << "Inputs: \n\n basisOrder: " << basisOrder << "\n sin(kx), k: " << k << "\n Number of nodes, h: : " << h << "\n Number of Gauss points per interval, gaussOrder: " << gaussOrder << " \n aconstant: " << aconstant << " and anorm: " << a << " \n\n xi = \n [" << setprecision(15) << x_i << "] \n\n xg = \n [" << setprecision(15) << x_g << "] \n\n  Exact equation @ x_g's u_ig= \n [" << setprecision(15) << u_ig << "\n @ nodes u_i = \n [" << setprecision(15) << u_i << "] \n\n Derivative of exact eq. @ x_g's du_ig = \n [" << setprecision(15) << du_ig << "] \n" << endl;
        
        writer << "Results: \n\n uh_RK = \n [" << setprecision(15) << uh_RK << "] \n\n duh_RK = \n [" << setprecision(15) << duh_RK << "] \n\n Error: \n L2diff = \n [" << setprecision(15) << L2diff << "] \n H1diff = \n [" << setprecision(15) << H1diff << "] \n L2 Norm RK: " << setprecision(15) << L2norm << "\n H1 Norm RK: " << setprecision(15) << H1norm << endl;
        
        writer.close(); //  Need to close the file string at the end every time
    }
    
    return 0;
}
