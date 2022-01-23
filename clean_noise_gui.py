import matplotlib.pyplot as plt
import numpy as np
from scipy import fft
from matplotlib.widgets import Slider, TextBox, Button
import warnings
warnings.filterwarnings("ignore")

class SignalParams:
    def __init__(self, A, f, phi):
        self.A = A
        self.f = f
        self.phi = phi

def create_signal(A, f, phi, t):
    return A*np.sin(2*np.pi*f*t+phi)
    
def create_noise(A, mu, sigma, size):
    return A * np.random.normal(mu, sigma, size=size)
    
def calc_PSD(x, fs):    
    Y = fft.fft(x)
    P = np.abs(Y)**2/x.size
    P_db = 10*np.log10(P)
    X = np.arange(0, P.size, fs/P.size)
    return X, Y, P_db
    
def generate_sample(Ts, tstart, tend, A_noise, *args):
    t = np.arange(tstart, tend, Ts)
    noise = A_noise*np.random.normal(0, 1, size=t.size)
    x = np.zeros(t.shape)
    for params in args:
        x = x + create_signal(params.A, params.f, params.phi, t)
        
    return t, x, noise
    
def clean_signal(X, Y, P_db, thresh):
    data_bins = P_db > thresh
    Y_cleaned = Y * data_bins
    cleaned_signal = fft.ifft(Y_cleaned)
    return cleaned_signal

def update_all(fig, axes, Ts, signal1, signal2, noise, thresh):
    fs = 1/Ts
    t, x, noise = generate_sample(Ts, 0, 0.2, noise, signal1, signal2)
    x_noised = x + noise
    X, Y, P_db = calc_PSD(x_noised, fs)
    x_cleaned = clean_signal(X, Y, P_db, thresh)
    X_end = int(fs/X.size * max_freq)
    
    fmin = min(signal1.f, signal2.f)
    dmax = 3/fmin
    t_index = int(dmax/Ts) + 1
    
    axes['ax1a'].clear()
    axes['ax1b'].clear()
    axes['ax2'].clear()
    axes['ax3'].clear()
    axes['ax4'].clear()
    
    axes['ax1a'].plot(t[:t_index], x[:t_index])
    axes['ax1b'].plot(t[:t_index], noise[:t_index])
    axes['ax2'].plot(t[:t_index], x_noised[:t_index])
    axes['ax3'].plot(X[:X.size//2][:X_end], P_db[:X.size//2][:X_end])
    axes['ax4'].plot(t[:t_index], x_cleaned[:t_index])
    
    axes['ax1a'].set_title("clean signal")
    axes['ax1b'].set_title("white noise")
    axes['ax2'].set_title("noisy signal")
    axes['ax3'].set_title("noisy PSD")
    axes['ax4'].set_title("cleaned signal")
    
    fig.canvas.draw_idle()

if __name__ == '__main__':
    fs = 50e3
    Ts = 1/fs
    min_freq = 200 # Hz
    max_freq = 5e3 # Hz
    default_noise = 2
    default_sig1 = SignalParams(1.2, 1300, np.pi/4)
    default_sig2 = SignalParams(0.8, 500, np.pi/3)
    default_thresh = 30

    fig = plt.figure()

    # adjust the main plot to make room for the sliders
    plt.subplots_adjust(left=0.3, bottom=0.2)
    
    # Make a horizontal slider to control the frequency.
    axfreq1 = plt.axes([0.25, 0.1, 0.65, 0.03])
    freq_slider1 = Slider(
        ax=axfreq1,
        label='Frequency 1[Hz]',
        valmin=min_freq,
        valmax=max_freq,
        valinit=default_sig1.f,
    )
    
    # Make a vertically oriented slider to control the amplitude
    axamp1 = plt.axes([0.1, 0.25, 0.0225, 0.63])
    amp_slider1 = Slider(
        ax=axamp1,
        label="Amplitude 1",
        valmin=0,
        valmax=10,
        valinit=default_sig1.A,
        orientation="vertical"
    )
    
    axfreq2 = plt.axes([0.25, 0.05, 0.65, 0.03])
    freq_slider2 = Slider(
        ax=axfreq2,
        label='Frequency 2[Hz]',
        valmin=min_freq,
        valmax=max_freq,
        valinit=default_sig2.f,
    )
    
    # Make a vertically oriented slider to control the amplitude
    axamp2 = plt.axes([0.15, 0.25, 0.0225, 0.63])
    amp_slider2 = Slider(
        ax=axamp2,
        label="Amplitude 2",
        valmin=0,
        valmax=10,
        valinit=default_sig2.A,
        orientation="vertical"
    )
    
    axnoise = plt.axes([0.2, 0.25, 0.0225, 0.63])
    noise_slider = Slider(
        ax=axnoise,
        label="Noise",
        valmin=0,
        valmax=10,
        valinit=default_noise,
        orientation="vertical"
    )
    
    axthresh = plt.axes([0.25, 0.25, 0.0225, 0.63])
    thresh_slider = Slider(
        ax=axthresh,
        label="Threshhold",
        valmin=0,
        valmax=50,
        valinit=default_thresh,
        orientation="vertical"
    )
        
    ax1a = fig.add_subplot(421)
    ax1b = fig.add_subplot(422)
    ax2 = fig.add_subplot(412)
    ax3 = fig.add_subplot(413)
    ax4 = fig.add_subplot(414)
    
    t, x, noise = generate_sample(Ts, 0, 0.2, default_noise, default_sig1, default_sig2)
    x_noised = x + noise
    X, Y, P_db = calc_PSD(x_noised, fs)
    x_cleaned = clean_signal(X, Y, P_db, default_thresh)
    X_end = int(fs/X.size * max_freq)
    
    fmin = min(default_sig1.f, default_sig2.f)
    dmax = 3/fmin
    t_index = int(dmax/Ts) + 1
    
    ax1a.plot(t[0:t_index], x[0:t_index])
    ax1b.plot(t[0:t_index], noise[0:t_index])
    ax2.plot(t[0:t_index], x_noised[0:t_index])
    ax3.plot(X[:X.size//2][:X_end], P_db[:X.size//2][:X_end])
    ax4.plot(t[:t_index], x_cleaned[:t_index])
    
    axes = {}
    axes['ax1a'] = ax1a
    axes['ax1b'] = ax1b
    axes['ax2'] = ax2
    axes['ax3'] = ax3
    axes['ax4'] = ax4
    
    ax1a.set_title("clean signal")
    ax1b.set_title("white noise")
    ax2.set_title("noisy signal")
    ax3.set_title("noisy PSD")
    ax4.set_title("cleaned signal")
    
    f = lambda _: update_all(fig,
                             axes,
                             Ts,
                             SignalParams(amp_slider1.val, freq_slider1.val, np.pi/4),
                             SignalParams(amp_slider2.val, freq_slider2.val, np.pi/3),
                             noise_slider.val,
                             thresh_slider.val
                           )
    
    freq_slider1.on_changed(f)
    freq_slider2.on_changed(f)
    amp_slider1.on_changed(f)
    amp_slider2.on_changed(f)
    noise_slider.on_changed(f)
    thresh_slider.on_changed(f)
    
    plt.show()