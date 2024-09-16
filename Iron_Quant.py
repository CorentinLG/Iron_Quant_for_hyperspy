import hyperspy.api as hs 
import numpy as np

def Iron_Quant(s, method = 'Voigt_fitting', auto_shift = False, mask=None, pre_quant = True, signal_range =(690., 703.), background_type='Polynomial'):

    """This function is meant to process Fe_L edge spectra in several steps:
        - remove background and yield the corresponding signal (s_back)
        - fit the peak around 710 to find the position of the maximum and shift it accordingly (Optional)
        - fit a double arctangente to normalize to the amount of Iron and yield the arctangent subtracted signal (s_arc)
        - apply calibration to quantify the Fe valency and yield the corresponding signal (Q1). If energy range > 727 eV, Second calibration is applied to the L2 edge as well (yield Q2)

    return : s_back, m, s_arc, Q1, Q2

    *** s : signal to process
    *** method : 
        - 'Integration' is based on direct spectral integration
        - 'Voigt_fitting' is based on fitting a number of Voigt functions that are contrained in position and width
    *** auto_shift : fit the peak around 710 to find the position of the maximum and shift it accordingly
    *** mask : to apply  a mask to the data to process
    *** pre_quant : If False, then the quantification is applied directly to the signal without first subtracting background and arctangent (return : Q1, Q2)"""

    import copy
    if pre_quant == True:
        s_back=s.remove_background(signal_range=signal_range, background_type=background_type,fast=False,)
        if auto_shift==True:
            for k in range (5):
                M = s_back.isig[707.:713.5].create_model()
                M.set_signal_range(709.7, 710.9)
                g= hs.model.components1D.Lorentzian()
                M.append(g)
                M.set_parameters_value('A',0.1, component_list=[g])
                M.set_parameters_value('centre',710.2, component_list=[g])
                M.set_parameters_value('gamma',0.5, component_list=[g])
                g.centre.value = 710.2
                g.centre.bmin = 709.5
                g.centre.bmax = 711.
                g.gamma.bmax = 0.9
                g.set_parameters_free(parameter_name_list=['A','centre', 'gamma'])
                M.multifit(bounded = True, mask = mask)
                s_back.axes_manager[0].axis = s_back.axes_manager[0].axis-(g.centre.value-710.)

        m=s_back.create_model()
        m.remove_signal_range(703.,716.5)
        m.remove_signal_range(718.,727.)
        m.extend([hs.model.components1D.Expression(
        expression="3.141592*height_1*arctan(3.141592*(x-x01)+3.141592/2)+3.141592*height_2*arctan(3.141592*(x-x02)+3.141592/2)-(3.141592*height_1*arctan(3.141592*(685-x01)+3.141592/2)+3.141592*height_2*arctan(3.141592*(685-x02)+3.141592/2))",
        name="double_arctan",
        height_1=0.01, 
        height_2=0.01,
        #x01=708.7, 
        x01=708.35,
        x02=722.45,
        )])

        m.set_parameters_not_free()
        m.set_parameters_free(parameter_name_list=['height_1'], component_list=['double_arctan'])
        m.set_parameters_free(parameter_name_list=['height_2'], component_list=['double_arctan'])
        m.multifit(mask = mask)

        s_arc = copy.deepcopy(s_back)
        if len(s.data.shape)==1:
            s_arc = s_back-(3.141592*m['double_arctan'].height_1.as_signal().data*np.arctan(3.141592*(s.axes_manager['Energy'].axis-m['double_arctan'].x01.as_signal().data)+3.141592/2)+
                3.141592*m['double_arctan'].height_2.as_signal().data*np.arctan(3.141592*(s.axes_manager['Energy'].axis-m['double_arctan'].x02.as_signal().data)+3.141592/2)-
                (3.141592*m['double_arctan'].height_1.as_signal().data*np.arctan(3.141592*(685-m['double_arctan'].x01.as_signal().data)+3.141592/2)+
                 3.141592*m['double_arctan'].height_2.as_signal().data*np.arctan(3.141592*(685-m['double_arctan'].x02.as_signal().data)+3.141592/2)))

        elif len(s.data.shape)==2:
            for j in range((s.data.shape[0])):
                s_arc.inav[j] = s_back.inav[j]-(3.141592*m['double_arctan'].height_1.as_signal().data[j]*np.arctan(3.141592*(s.axes_manager['energy'].axis-m['double_arctan'].x01.as_signal().data[j])+3.141592/2)+
                3.141592*m['double_arctan'].height_2.as_signal().data[j]*np.arctan(3.141592*(s.axes_manager['Energy'].axis-m['double_arctan'].x02.as_signal().data[j,i])+3.141592/2)-
                (3.141592*m['double_arctan'].height_1.as_signal().data[j]*np.arctan(3.141592*(685-m['double_arctan'].x01.as_signal().data[j])+3.141592/2)+
                3.141592*m['double_arctan'].height_2.as_signal().data[j]*np.arctan(3.141592*(685-m['double_arctan'].x02.as_signal().data[j])+3.141592/2)))

        elif len(s.data.shape) ==3:
            for i in range ((s.data.shape[1])):
                for j in range((s.data.shape[0])):
                    s_arc.inav[i,j] = s_back.inav[i,j]-(3.141592*m['double_arctan'].height_1.as_signal().data[j,i]*np.arctan(3.141592*(s.axes_manager['Energy'].axis-m['double_arctan'].x01.as_signal().data[j,i])+3.141592/2)+
                                                        3.141592*m['double_arctan'].height_2.as_signal().data[j,i]*np.arctan(3.141592*(s.axes_manager['Energy'].axis-m['double_arctan'].x02.as_signal().data[j,i])+3.141592/2)-
                                                        (3.141592*m['double_arctan'].height_1.as_signal().data[j,i]*np.arctan(3.141592*(685-m['double_arctan'].x01.as_signal().data[j,i])+3.141592/2)+
                                                         3.141592*m['double_arctan'].height_2.as_signal().data[j,i]*np.arctan(3.141592*(685-m['double_arctan'].x02.as_signal().data[j,i])+3.141592/2)))
    else : s_arc = s
    
    if method == 'Voigt_fitting':
        pic_pos_list=[705.4, 706.3, 707.3, 707.9, 708.45, 710., 711.2, 712.4]
        fhwm_list=[0.2,0.3, 0.3,0.3,0.3,0.3,0.3,0.3]
        gamma_list=[0.3,0.3, 0.35,0.35,0.35,0.45,0.35,0.3]
    
        mf=s_arc.isig[705.:713.].create_model()
    
        f=[]
        for j in range (len(pic_pos_list)):
            f.append(hs.model.components1D.Voigt())
            mf.append(f[j])
            mf.set_parameters_value('area', 0.05, component_list=[f[j]])
            mf.set_parameters_value('centre',pic_pos_list[j], component_list=[f[j]])
            mf.set_parameters_value('FWHM',0.3, component_list=[f[j]])
            mf.set_parameters_value('gamma',0.1, component_list=[f[j]])

            f[j].centre.bmin = pic_pos_list[j]-0.25
            f[j].centre.bmax = pic_pos_list[j]+0.25
            f[j].FWHM.bmax = fhwm_list[j]
            f[j].FWHM.bmin = 0
            f[j].area.bmin = 0
            f[j].gamma.bmin = 0
            f[j].gamma.bmax = gamma_list[j]

            f[j].set_parameters_free(parameter_name_list=['area','centre', 'gamma', 'FWHM'])
        
        mf.multifit(optimizer = 'lm', bounded = True, mask = mask)
        
        a= 0.00796570363521859
        b=-0.07362046487532675
        Int = f[5].area.as_signal()/(f[1].area.as_signal()+f[2].area.as_signal()+f[3].area.as_signal()+f[4].area.as_signal()+f[5].area.as_signal())
        Q = (Int+b)/a
        if pre_quant == True:
            return s_back, s_arc, m, mf, Q
        elif pre_quant==False:
            return mf, Q, Int
    
    elif method == 'Integration':
        Int_Fe3 = s_arc.isig[709.5:712.].integrate1D(-1)/s_arc.isig[706.:712.].integrate1D(-1) # Best bet after reassesment

        # Calibration values based on fitted shift position and extra standards included (as of March 2022, st dev = 4.5)
        a = 0.006058150502673548
        b= -0.13718198657527195

        Q1 = (Int_Fe3+b)/a
        Q2=copy.deepcopy(Q1)
        if s.axes_manager['Energy'].axis.any()>727.==True:
            Int_Fe3_b = (s_arc.isig[722.:727.].integrate1D(-1)/s_arc.isig[718.:727.].integrate1D(-1))
            Q2 = (Int_Fe3_b-0.18553696661)/0.0066369293    
        if pre_quant == True:
            return s_back, m, s_arc, Q1, Q2
        elif pre_quant==False:
            return Q1, Q2
