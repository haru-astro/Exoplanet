from astropy import units as u
from astropy.coordinates import SkyCoord
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import statistics

#このデータは１次処理が済んでいる
#compと比較し、シンチレーションを補正すればOK
#シンチレーション補正はどう扱う？
#S/N比を求めるため？
'''
toi519 = pd.read_table('TOI-519/TOI-519_muscat_z_target.dat',sep=r'\s+')
comp = pd.read_table('TOI-519/TOI-519_muscat_z_comp.dat',sep=r'\s+')

time_J=toi519.iloc[:,0]
flux=toi519.iloc[:,1]
error=toi519.iloc[:,2]
air_mass=toi519.iloc[:,3]
sky=toi519.iloc[:,4]

comp_flux=comp.iloc[:,1]
comp_error=comp.iloc[:,2]
comp_air_mass=comp.iloc[:,3]
comp_sky=comp.iloc[:,4]

time=[]
for i in time_J:
    time.append(i-8818)

re_flux=flux/comp_flux

error_re_flux=[]
for i in range(len(flux)):
    error_re_flux.append(math.sqrt((error[i]/comp_flux[i])**2+(comp_error[i]*flux[i])**2/(comp_flux[i])**4))

#トランジットしていないところの平均を取る
#ここの値は適宜変更！
re_flux_no=list(re_flux)
del re_flux_no[92:162]
flux_no=list(flux)
del flux_no[92:162]

#平均、規格化する用
ave_re_flux=sum(re_flux_no)/len(re_flux_no)
ave_flux=sum(flux_no)/len(flux_no)
ave_comp_flux=sum(comp_flux)/len(comp_flux)

#規格化
std_error_re_flux = [n/ave_re_flux for n in error_re_flux]
std_error_flux = [n/ave_flux for n in error]
std_error_comp_flux = [n/ave_comp_flux for n in comp_error]

std_re_flux = [n/ave_re_flux for n in re_flux]
std_flux = [n/ave_flux for n in flux]
std_comp_flux = [n/ave_comp_flux for n in comp_flux]

data=[]
for i in range (220):
    l=[time_J[i],std_re_flux[i],std_error_re_flux[i],air_mass[i]]
    data.append(l)
columns=["GJD-2450000","Flux","error","airmass"]
filename="TOI-519_data.dat"
data_with_headers = np.vstack([columns, data])
np.savetxt(filename, data_with_headers, delimiter=' ', fmt='%s')


#plt.errorbar(time,std_re_flux, yerr = std_error_re_flux, capsize=5, fmt='o', markersize=3, ecolor='black', markeredgecolor = "black", color='w')
#plt.scatter(time,std_re_flux, s=5, c=None, marker='o')
#plt.errorbar(time,std_flux, yerr = std_error_flux, capsize=5, fmt='o', markersize=3, ecolor='black', markeredgecolor = "black", color='w')
#plt.scatter(time,std_flux, s=5, c=None, marker='o')
#plt.errorbar(time,std_comp_flux, yerr = std_error_comp_flux, capsize=5, fmt='o', markersize=3, ecolor='black', markeredgecolor = "black", color='w')
#plt.scatter(time,std_comp_flux, s=5, c=None, marker='o')

#plt.xlabel('GJD-2458818[day]')
#plt.ylabel('relative_flux')
#plt.ylim(0.89,1.01)
#plt.xticks(np.arange(-0.5, 0.50001, step=0.1))
#plt.yticks(np.arange(-0.5, 0.50001, step=0.1))
#plt.grid()
#plt.savefig("TOI-519_transit.png", format="png", dpi=300)
#plt.title("TOI-519 relative_flux")
#plt.show()
'''

#ここからフィッティング

import numpy as np
import matplotlib.pyplot as plt
import pytransit
import emcee
from scipy.optimize import minimize
from scipy.stats import norm
import corner
import matplotlib.gridspec as gridspec
from multiprocessing import Pool
from ldtk import LDPSetCreator, BoxcarFilter

# データの設定
data=pd.read_table('TOI-519/TOI-519_data.dat',sep='\s+')
flux = data.iloc[:,1]  # フラックスデータ
flux_err = data.iloc[:,2]  # 誤差
time=data.iloc[:,0]-8818

# Fixed Params
p=1.26523

# degree
degree=2

# トランジットモデル
model = pytransit.QuadraticModel()
model.set_data(time)

# リム暗化係数理論値
# ここを追加
filters = [
            # BoxcarFilter('g', 400, 550),
            # BoxcarFilter('r', 550, 700),
            # BoxcarFilter('i', 700, 820),
            BoxcarFilter('z_s', 820, 920),
          ]

## Stellar parameters
sc = LDPSetCreator(teff=(3322,   49), # effective temperature
                   logg=(4.87,  0.03), # surface gravity
                      z=(0.27, 0.09), # metallicity
                     filters=filters)

ps = sc.create_profiles()                # Create the limb darkening profiles
qc,qe = ps.coeffs_qd(do_mc=True, n_mc_samples=10000)         # Estimate quadratic law coefficients

# ログ尤度関数を定義
def log_likelihood(params):
    b, k, ln_a, t0, u1, u2= params
    a = np.exp(ln_a)
    ldc = np.array((u1, u2))

    if b < 0 or b+k >= 1 or a<=0 or b/a>=1 or k<=0 or u1<=0 or u2<=0 or u1+u2>=1:
        return -np.inf
    else:
        i = math.acos(b/a)
        model_flux_tr = model.evaluate_ps(k,ldc,t0,p,a,i,0,0) #この0は何？
        c = np.polyfit(time-t0, flux/model_flux_tr, degree) 
        model_flux = model_flux_tr * np.polyval(c, time-t0)
        ll=(-0.5*np.sum( (flux - model_flux)**2 / flux_err**2 ) - 0.5*np.sum(np.log(2*np.pi*flux_err**2) ))
        return ll
    

# ログ事前分布を定義
def log_prior(params):
    b, k, ln_a, t0, u1, u2= params
    lp=0
    err_scale_factor = 10
    lp += -0.5 * (u1 - qc[0][0])**2 / (qe[0][0]*err_scale_factor)**2
    lp += -0.5 * (u2 - qc[0][1])**2 / (qe[0][1]*err_scale_factor)**2
    return lp

# ログ事後分布(積)を定義
def log_posterior(params):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    #事後分布は尤度関数と事前分布の積、対数を取っているので和にする
    return lp + log_likelihood(params)

# ログ事後分布(積)のマイナス
def neg_log_posterior(p):    
    return -1*log_posterior(p)

# Initial_params
# 福井先生の結果から引用
initial_params = [0.244, 0.297, 2.317, 0.304, 0.289, 0.353]
n_walkers = 32  # ウォーカーの数
n_dim = len(initial_params)  # パラメータの数
n_steps = 10000

# 全体の確率で割る前の値
res = minimize(neg_log_posterior, initial_params, method='Nelder-Mead')

pos = [res.x + 1e-6*np.random.randn(n_dim) for i in range(n_walkers)]
sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_posterior)

# MCMC実行
sampler.run_mcmc(pos, n_steps, progress=True)

# discardなどの設定
discard=5000
thin=10

# サンプル結果(全体の確率で割る前)の取得
samples = sampler.get_chain(discard=discard, thin=thin, flat=True)

#尤度関数(全体の確率で割る前)の変化
#事後分布はこれを全体の確率で割っているだけ

log_prob = sampler.get_log_prob(flat=True, discard=discard, thin=thin)

'''
#これやる場合はdiscardの設定消してね
plt.figure(figsize=(8,4))
plt.rcParams['font.size']=12
for i in range(n_walkers):
    plt.plot(log_prob[:,i],alpha=0.1)
plt.xlabel('Number of steps', fontsize=20)
plt.ylabel('Log probability', fontsize=20)
plt.show()
'''

# 各パラメータの中央値と68%範囲
for i in range(6):
    med = np.percentile(samples[:, i], [16, 50, 84])
    q = np.diff(med)
    print(med[1], "+",q[1], "-", q[0])

#print("b:", b_mcmc)
#print("k:", k_mcmc)
#print("ln_a:", ln_a_mcmc)
#print("t0:", t0_mcmc)
#print("u1", u1_mcmc)
#print("u2:", u2_mcmc)

#最尤値の最大値とそれをとる値
argmax = np.argmax(log_prob)
bestp = samples[argmax,:]

b_best, k_best, ln_a_best, t0_best, u1_best, u2_best = bestp[0:6]
print(bestp[0:6])

a_best = np.exp(ln_a_best) # a/Rs
i_best = math.acos(b_best/a_best)
ldc_best = np.array((u1_best, u2_best))

model_flux_tr = model.evaluate_ps(k_best, [u1_best, u2_best], t0_best, p, a_best, i_best,0,0) 
c = np.polyfit(time-t0_best, flux/model_flux_tr, degree)
model_flux = model_flux_tr * np.polyval(c, time-t0_best)

print("尤度関数:", log_likelihood(bestp[0:6]))
print("BIC:", -2*log_likelihood(bestp[0:6])+(degree+7)*math.log(len(flux)))


#フィットの結果を表示
fig = plt.figure(figsize=(10,8))
gs = gridspec.GridSpec(2,1, height_ratios=(4, 1))
ax1 =  fig.add_subplot(gs[:1, 0], visible=False)
ax2 =  fig.add_subplot(gs[-1, 0], visible=False)

# 上のプロット
ax1 = fig.add_subplot(2,1,1) 
ax1.errorbar(time, flux, yerr=flux_err, capsize=5, fmt='o', markersize=3, ecolor='black', markeredgecolor='black', color='w')
ax1.plot(time, model_flux, label='Model Fit', color='C1')
ax1.set_title('TOI-519 Transit Light Curve with Model Fit')
ax1.set_ylabel('Flux (normalized)')
ax1.legend()
ax1.grid()

# 下のプロット（残差表示）
ax2 = fig.add_subplot(2,1,2) 
ax2.errorbar(time, flux - model_flux, yerr=flux_err, capsize=5, fmt='o', markersize=3, ecolor='black', markeredgecolor='black', color='w')
plt.axhline(0, color='C1')   
ax2.set_xlabel('GJD-2458818 [day]')
ax2.set_ylabel('Residuals')
ax2.grid()

#plt.tight_layout()
#plt.savefig("TOI-519_Fitting_deg1.png", format="png", dpi=300)
#plt.show()

#コーナープロット
#fig = corner.corner(samples, labels=["b", "k", "ln_a", "t0", "u1", "u2"],quantiles=[0.16,0.5,0.84],show_titles=True, title_fmt='.3f', label_kwargs={'fontsize':20})
#plt.savefig("TOI-519_cornerplot_deg1.png", format="png", dpi=300)
#plt.show()






'''
#下のはexoplanetを使おうとして失敗したもの
star = starry.Primary(starry.Map(udeg=2), m=1.0)  # 恒星のモデル (2次のlimb-darkening)
planet = starry.Secondary(starry.Map(), r=0.1, porb=1.27, t0=0.0)  # 惑星のモデル
sys = starry.System(star, planet)

# PyMCでモデルを定義
with pm.Model() as model:
    # 惑星パラメータの事前分布を定義
    period = pm.Normal("period", mu=1.27, sigma=0.01)  # TOI-519の周期
    t0 = pm.Normal("t0", mu=0.31, sigma=0.1)  # 中心通過時間
    r_star = pm.Normal("r_star", mu=0.54, sigma=0.05)  # 恒星TOI-519の半径
    r_planet = pm.Normal("r_planet", mu=0.11, sigma=0.05)  # 惑星の半径（恒星半径に対する比）

    # 軌道モデルを生成
    orbit = KeplerianOrbit(period=period, t0=t0)

    # トランジットモデルの生成（starryを使用）
    light_curve = sys.flux(t=time)

    # 観測データとトランジットモデルの誤差を考慮
    flux_model = pm.Normal("flux_model", mu=light_curve, sigma=std_error_re_flux, observed=flux)

    # サンプリング実行
    trace = pm.sample(tune=1000, draws=1000, return_inferencedata=True)
# フィット結果の確認とプロット
with model:
    pm.plot_trace(trace)
    plt.show()

# トランジットモデルのプロット
plt.plot(time, std_re_flux, "k.", label="Data")
for sample in trace.posterior["r_planet"].values[::10]:
    orbit = KeplerianOrbit(period=trace.posterior["period"].mean(), t0=trace.posterior["t0"].mean())
    light_curve = xo.StarryLightCurve(u=[0.3, 0.2]).get_light_curve(orbit=orbit, r=sample, t=time)
    plt.plot(time, light_curve[:, 0], alpha=0.3)
plt.xlabel("Time [days]")
plt.ylabel("Normalized Flux")
plt.legend()
plt.show()
'''