
clear; clc; close all;

%% Common settings
colors = [0.00 0.45 0.74;
          0.85 0.32 0.10;
          0.47 0.67 0.19;
          0.55 0.25 0.64];

styles = {'--','-.',':','-'};
red = [0.86 0.10 0.12];

%% Left and Right Bound
a = -1;
b = 1;

gamma_plot = [1, 0.1, 0.01, 0.001];
gammas = logspace(0, -3, 60);

% Fine spatial grid
x = linspace(-4, 4, 20001);
dx = x(2) - x(1);

% Potential and unnormalized density
f = @(x) x.^2 + x.^4/8 - x;
w = exp(-f(x));

% Distance to K = [a,b]
distK = max(a-x, 0) + max(x-b, 0);

%% Constrained target density
p = w .* (x >= a & x <= b);
p = p / trapz(x, p);

% Target CDF
F = cumtrapz(x, p);
F = F / F(end);

%% Quantile grid for computing W2
t = linspace(1e-7, 1-1e-7, 100000);

% Unique CDF values for inverse interpolation
[F_unique, idx] = unique(F, 'stable');
x_unique = x(idx);

Finv = interp1(F_unique, x_unique, t, ...
               'linear', 'extrap');

%% Compute W2 for different gamma
W2 = zeros(size(gammas));

for k = 1:length(gammas)
    gamma = gammas(k);

    % Penalized density
    q = w .* exp(-distK.^2/(2*gamma));
    q = q / trapz(x, q);

    % Penalized CDF
    G = cumtrapz(x, q);
    G = G / G(end);

    [G_unique, idx] = unique(G, 'stable');
    xG = x(idx);

    Ginv = interp1(G_unique, xG, t, ...
                   'linear', 'extrap');

    % One-dimensional W2
    W2(k) = sqrt(trapz(t, (Finv-Ginv).^2));
end

%% 1 x 2 Figure
% Extra vertical room for two-line subcaption
fig = figure('Color','w','Units','inches', ...
    'Position',[1 1 15 5.2], ...
    'PaperPositionMode','auto');

%% Left subplot: densities
ax1 = axes('Parent',fig,'Units','normalized', ...
    'Position',[0.065 0.32 0.42 0.61]);
hold(ax1,'on');

plot(ax1, x, p, 'k-', 'LineWidth', 1.2, ...
    'DisplayName', '\pi(x)');

for k = 1:length(gamma_plot)
    gamma = gamma_plot(k);

    q = w .* exp(-distK.^2/(2*gamma));
    q = q / trapz(x, q);

    plot(ax1, x, q, ...
        'Color',colors(k,:), ...
        'LineStyle',styles{k}, ...
        'LineWidth',1.10, ...
        'DisplayName', ...
        sprintf('\\pi^\\gamma(x), \\gamma= %.3g',gamma));
end

xline(ax1,a,'k:', 'HandleVisibility','off');
xline(ax1,b,'k:', 'HandleVisibility','off');

xlabel(ax1,'$x$', 'Interpreter','latex');
ylabel(ax1,'Density', 'Interpreter','latex');

legend(ax1,'Location','northwest');
xlim(ax1,[-2.5 2.5]);

grid(ax1,'on');
box(ax1,'on');
set(ax1,'FontSize',12);

%% Right subplot: Wasserstein distance
ax2 = axes('Parent',fig,'Units','normalized', ...
    'Position',[0.575 0.32 0.39 0.61]);
hold(ax2,'on');

semilogx(ax2,gammas,W2,'b-', ...
    'LineWidth',1.2);

% % Reference O(sqrt(gamma))
% C = W2(end)/sqrt(gammas(end));
% 
% semilogx(ax2,gammas,C*sqrt(gammas), ...
%     'r--','LineWidth',1.2);

xlabel(ax2,'$\gamma$','Interpreter','latex');
ylabel(ax2,'$W_2(\pi,\pi^\gamma)$', ...
       'Interpreter','latex');

legend(ax2, ...
    '$W_2(\pi,\pi^\gamma)$', ...
    '$C\sqrt{\gamma}$', ...
    'Interpreter','latex', ...
    'Location','northeast');

% Explicitly enforce logarithmic gamma axis
set(ax2, ...
    'XScale', 'log', ...
    'XDir', 'reverse', ...
    'XLim', [1e-3 1], ...
    'XTick', [1e-3 1e-2 1e-1 1], ...
    'FontSize', 12);

grid(ax2,'on');
box(ax2,'on');

%% LaTeX subcaptions below plots

% (a) Two-line subcaption
% Each cell in String becomes a separate line.
captionA = { ...
    ['(a) Target constrained density ' ...
     '$\pi(x)\propto e^{-x^2-\frac{x^4}{8}+x}' ...
     '\mathbf{1}_{\mathcal{K}}(x)$;'], ...
    ['Approximated density ' ...
     '$\pi^\gamma(x)' ...
     '\propto e^{-x^2-\frac{x^4}{8}+x-' ...
     '(2\gamma)^{-1}\|x-\mathcal{P}_{\mathcal{K}}(x)\|^2}$'] ...
    };

annotation(fig,'textbox', ...
    [0.025 0.07 0.50 0.15], ...
    'String',captionA, ...
    'Interpreter','latex', ...
    'FontSize',18, ...
    'HorizontalAlignment','center', ...
    'VerticalAlignment','middle', ...
    'EdgeColor','none', ...
    'FitBoxToText','off');

% (b) Wasserstein subcaption
annotation(fig,'textbox', ...
    [0.575 0.10 0.39 0.08], ...
    'String','(b) 2-Wasserstein Distance', ...
    'Interpreter','latex', ...
    'FontSize',18, ...
    'HorizontalAlignment','center', ...
    'VerticalAlignment','middle', ...
    'EdgeColor','none', ...
    'FitBoxToText','off');

%% Display numerical results
fprintf('     gamma          W2\n');
fprintf('--------------------------\n');

for k = 1:length(gammas)
    fprintf('%12.6f    %12.8f\n',gammas(k),W2(k));
end
