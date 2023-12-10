import math
import matplotlib.pyplot as plt
import numpy as np

AntGainBS = 21 # Коэффициент усиления антенны BS, дБм
N_p = 15 # Запас мощности сигнала на проникновения сквозь стены, дБ
MIMO_Gain = 3*2 # Число приемо-передающих антенн на BS
f = 1.8 # Диапазон частот, ГГц

# Расчет бюджета восходящего канала:
IM = 1 # Запас мощности сигнала на интерференцию, дБ
Feeder_Loss = 2 # дБ
Noise_f_bs = 2.4 # Коэффициент шума приемника BS, дБ
SINR_UL = 4 # Требуемое отношение SINR для UL, дБ
BW_UL = 10*10**6 # Полоса частот в UL, Гц
TX_POW_UE = 24 # Мощность передатчика пользовательского терминала UE, дБм
ThermalNoise_UL = -174 + 10 * math.log10(BW_UL)
RxSens_bs = Noise_f_bs + ThermalNoise_UL + SINR_UL
MAPL_UL = (RxSens_bs - TX_POW_UE + Feeder_Loss - AntGainBS - MIMO_Gain + IM + N_p) * (-1)
print('budjet voshodyashego', MAPL_UL ,'dB')

# Расчет бюджета нисходящего канала:
TX_BOW_BS = 46 # Мощность передатчиков BS, дБм
BW_DL = 20*10**6 # Полоса частот в DL, Гц
ThermalNoise_DL = -174 + 10 * math.log10(BW_DL)
RxSens_ue = -98
MAPL_DL = (-RxSens_ue + TX_BOW_BS - Feeder_Loss + AntGainBS + MIMO_Gain - IM - N_p)
print('bugjet nishodyashego', MAPL_DL ,'dB')

# UMiNLOS
d = np.arange(1, 3000) # м
PL_UM = 26 * np.log10(f)+ 22.7 + 36.7 * np.log10(d)
intersection = np.intersect1d(MAPL_UL, MAPL_DL)
Sum_UM = round(4000**2 / (1.95 * 574**2))
print('UMiNLOS R (UL)=', 574 ,'m')
print('UMiNLOS R (DL)=', 1710 ,'m')
print('UMiNLOS kol-vo bazovyh stansiy: ', Sum_UM)

# COST231
hms = 5
a_U = 3.2 * np.log10(11.75 * hms) * 2 - 4.97
a_R = (1.1* np.log10(f)) * hms * (1.56 *np.log10(f) - 0.8)
Lclutter_SU = - (2 * (np.log10(f/28))**2 + 5.4)
s = 44.9 - 6.55 * np.log10(f)
hBS = 50
PL_COST = 69.55 + 26.16 * np.log10(f) - 13.8 * np.log10(hBS) - a_R + s * np.log10(d) + Lclutter_SU
Sum_CO = round(100000**2 / (1.95 * 295**2))
print('COST231 R (UL) =', 295 ,'m')
print('COST231 R (DL) =', 742 ,'m')
print('COST231 kol-vo bazovyh snatsyi: ', Sum_CO)

# Davidson
f1 = 0.8
d1 = np.arange(1, 3000)
L_Hata = 69.55+26.16 * np.log(f1)-13.82*np.log(10)-a_U+(44.9+6.55*np.log(10)*np.log(d))
L_HD = L_Hata - 4.7 * np.abs(np.log(6.2/d1))*((10-300)/600) - ((f1/250)*np.log(1500/f1))

# Walfish-Ikegami
Walfish = 42.6 + 20 * np.log10(f) + 26 * np.log10(d)
Walfish_L = 33.44 + 20 * np.log10(f) + 20 * np.log10(d)

plt.figure(figsize=(10, 6))
plt.plot(d, L_HD, label='Davidson')
plt.plot(d, PL_UM, label='Path Loss UMiNLOS')
plt.plot(d, PL_COST, label='Path Loss COST')
plt.plot(d, Walfish, label='Walfish-Ikegami Pryamaya v')
plt.plot(d, Walfish_L, label='Walfish-Ikegami Otsutstvie v')
plt.axhline(y=MAPL_UL, color='r', linestyle='--', label='MAPL_UL')
plt.axhline(y=MAPL_DL, color='g', linestyle='--', label='MAPL_DL')
plt.xlabel('Distance (m)')
plt.ylabel('Path Loss (dB)')
plt.title('Path Loss vs. Distance')
plt.legend()
plt.grid(True)
plt.show()