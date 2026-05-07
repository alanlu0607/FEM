%% FEM vs FDM on L-shaped domain
% Solves -Laplacian(u) = 1, u=0 on boundary

clear; clc; close all;

inL = @(xi,yi) ~(xi > 0 & yi > 0);

%% =========================================================
%  Single-resolution visualization (FDM N=40, FEM M=40)
%% =========================================================
N = 40; M = 40;
[u_fdm, x, y] = solve_fdm(N, inL);
[u_fem_vec, nodeCoords, tria] = solve_fem(M, inL);
[Xg, Yg] = meshgrid(x, y);

figure('Position',[100 100 1400 450]);

subplot(1,3,1);
u_plot = u_fdm';
u_plot(~inL(Xg,Yg)) = NaN;
surf(Xg, Yg, u_plot, 'EdgeColor', 'none'); %[0.3 0.3 0.3]);
colorbar; colormap(jet); view(120,30);
title('FDM Solution'); xlabel('x'); ylabel('y'); zlabel('u');

subplot(1,3,2);
trisurf(tria, nodeCoords(:,1), nodeCoords(:,2), u_fem_vec, ...
    'EdgeColor', 'none', 'FaceColor','interp');
colorbar; colormap(jet); view(120,30);
title('FEM Solution'); xlabel('x'); ylabel('y'); zlabel('u');

subplot(1,3,3);
u_fem_interp = griddata(nodeCoords(:,1), nodeCoords(:,2), u_fem_vec, Xg, Yg);
diff_plot = abs(u_fdm' - u_fem_interp);
diff_plot(~inL(Xg,Yg)) = NaN;
surf(Xg, Yg, diff_plot, 'EdgeColor','none'); %[0.3 0.3 0.3]);
colorbar; colormap(jet); view(60,30);
title('|FDM - FEM|'); xlabel('x'); ylabel('y');

fprintf('FDM max u = %.6f\n', max(u_fdm(~isnan(u_fdm))));
fprintf('FEM max u = %.6f\n', max(u_fem_vec));
fprintf('Max pointwise difference = %.2e\n', max(diff_plot(~isnan(diff_plot))));

%% =========================================================
%  Mesh Progression at Multiple Resolutions
%% =========================================================
Ns_vis = [10, 20, 40, 80, 160];
Ms_vis = Ns_vis; % [5,  10, 20, 40, 80 ];
nVis = length(Ns_vis);

figure('Position', [100 100 1800 650]);
sgtitle('Mesh Progression: FDM (top) vs FEM (bottom)');

for k = 1:nVis
    % --- FDM row (top) ---
    subplot(2, nVis, k);
    [uk, xk, yk] = solve_fdm(Ns_vis(k), inL);
    [Xk, Yk] = meshgrid(xk, yk);
    up = uk';
    up(~inL(Xk, Yk)) = NaN;
    if k <= 3
        surf(Xk, Yk, up, 'EdgeColor', [0.1 0.1 0.1]);
    else
        surf(Xk, Yk, up, 'EdgeColor', 'none');
    end
    view(120, 30); colormap(jet);
    title(sprintf('FDM  N=%d', Ns_vis(k)));
    xlabel('x'); ylabel('y'); zlabel('u');

    % --- FEM row (bottom) ---
    subplot(2, nVis, nVis + k);
    [uk, coords, triak] = solve_fem(Ms_vis(k), inL);
    if k <= 3
        trisurf(triak, coords(:,1), coords(:,2), uk, ...
        'EdgeColor', [0.1 0.1 0.1], 'FaceColor', 'interp');
    else
        trisurf(triak, coords(:,1), coords(:,2), uk, ...
        'EdgeColor', 'none', 'FaceColor', 'interp');
    end
    view(120, 30); colormap(jet);
    title(sprintf('FEM  M=%d', Ms_vis(k)));
    xlabel('x'); ylabel('y'); zlabel('u');
end

%% =========================================================
%  Convergence Analysis vs Reference Solution
%% =========================================================
N_ref = 600;
fprintf('\nComputing reference solution (FDM, N=%d)...\n', N_ref);
[u_ref, x_ref, y_ref] = solve_fdm(N_ref, inL);
[Xr, Yr] = meshgrid(x_ref, y_ref);
mask_ref = inL(Xr, Yr);

Ns = [10, 20, 40, 80, 160];
Ms = Ns; % [5,  10, 20, 40, 80 ];
h_fdm = 2 ./ (Ns - 1);
h_fem = 2 ./ Ms;
err_fdm = zeros(size(Ns));
err_fem = zeros(size(Ms));

fprintf('\nFDM convergence:\n');
for k = 1:length(Ns)
    [uk, xk, yk] = solve_fdm(Ns(k), inL);
    [Xk, Yk] = meshgrid(xk, yk);
    % interp2 needs Z in (row=y, col=x) orientation, hence uk'
    u_interp = interp2(Xk, Yk, uk', Xr, Yr, 'linear', NaN);
    d = u_interp - u_ref';
    err_fdm(k) = sqrt(nanmean(d(mask_ref).^2));
    fprintf('  N=%3d  h=%.4f  err=%.3e\n', Ns(k), h_fdm(k), err_fdm(k));
end

fprintf('\nFEM convergence:\n');
for k = 1:length(Ms)
    [uk, coords, ~] = solve_fem(Ms(k), inL);
    u_interp = griddata(coords(:,1), coords(:,2), uk, Xr, Yr);
    d = u_interp - u_ref';
    err_fem(k) = sqrt(nanmean(d(mask_ref).^2));
    fprintf('  M=%3d  h=%.4f  err=%.3e\n', Ms(k), h_fem(k), err_fem(k));
end

% Log-log linear regression
p_fdm = polyfit(log(h_fdm), log(err_fdm), 1);
p_fem = polyfit(log(h_fem), log(err_fem), 1);

figure;
loglog(h_fdm, err_fdm, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 8); hold on;
loglog(h_fem, err_fem, 'r-s', 'LineWidth', 1.5, 'MarkerSize', 8);
h_range = linspace(min([h_fdm, h_fem]), max([h_fdm, h_fem]), 50);
loglog(h_range, exp(polyval(p_fdm, log(h_range))), 'b--', 'LineWidth', 1);
loglog(h_range, exp(polyval(p_fem, log(h_range))), 'r--', 'LineWidth', 1);
xlabel('h (mesh spacing)');
ylabel('L^2 error (vs reference)');
legend(sprintf('FDM (slope=%.2f)', p_fdm(1)), ...
       sprintf('FEM (slope=%.2f)', p_fem(1)), ...
       'FDM fit', 'FEM fit', 'Location', 'northwest');
title(sprintf('Convergence vs Reference Solution (FDM  N=%d)', N_ref));
grid on;

fprintf('\nFitted convergence rates:\n');
fprintf('  FDM: O(h^%.2f)\n', p_fdm(1));
fprintf('  FEM: O(h^%.2f)\n', p_fem(1));

%% =========================================================
%  LOCAL FUNCTIONS
%% =========================================================

function [u_2d, x, y] = solve_fdm(N, inL)
    h = 2/(N-1);
    x = linspace(-1, 1, N);
    y = linspace(-1, 1, N);

    idx = zeros(N,N);
    count = 0;
    for j = 1:N
        for i = 1:N
            if inL(x(i), y(j))
                count = count + 1;
                idx(i,j) = count;
            end
        end
    end
    nDOF = count;

    A = spalloc(nDOF, nDOF, 5*nDOF);
    b = zeros(nDOF, 1);

    for j = 1:N
        for i = 1:N
            if ~inL(x(i), y(j)), continue; end
            k = idx(i,j);

            % Check grid edges first, then reentrant corner neighbors
            isBnd = (i==1) || (i==N) || (j==1) || (j==N);
            if ~isBnd
                isBnd = ~inL(x(i-1),y(j)) || ~inL(x(i+1),y(j)) || ...
                        ~inL(x(i),y(j-1)) || ~inL(x(i),y(j+1));
            end

            if isBnd
                A(k,k) = 1; b(k) = 0;
            else
                A(k,k) = 4/h^2;
                A(k, idx(i-1,j)) = -1/h^2;
                A(k, idx(i+1,j)) = -1/h^2;
                A(k, idx(i,j-1)) = -1/h^2;
                A(k, idx(i,j+1)) = -1/h^2;
                b(k) = 1;
            end
        end
    end

    u_vec = A \ b;

    u_2d = nan(N,N);
    for j = 1:N
        for i = 1:N
            if inL(x(i),y(j))
                u_2d(i,j) = u_vec(idx(i,j));
            end
        end
    end
end

function [u_vec, nodeCoords, tria] = solve_fem(M, inL)
    xn = linspace(-1,1,M+1);
    yn = linspace(-1,1,M+1);

    nodeCoords = [];
    nodeIdx = zeros(M+1,M+1);
    nCount = 0;
    for j = 1:M+1
        for i = 1:M+1
            xi = xn(i); yi = yn(j);
            if inL(xi,yi)
                nCount = nCount+1;
                nodeCoords(nCount,:) = [xi, yi]; %#ok
                nodeIdx(i,j) = nCount;
            end
        end
    end
    nNodes = nCount;

    tria = [];
    for j = 1:M
        for i = 1:M
            c = [nodeIdx(i,j), nodeIdx(i+1,j), nodeIdx(i+1,j+1), nodeIdx(i,j+1)];
            if all(c > 0)
                tria(end+1,:) = [c(1), c(2), c(3)]; %#ok
                tria(end+1,:) = [c(1), c(3), c(4)]; %#ok
            end
        end
    end
    nElem = size(tria,1);

    K = spalloc(nNodes, nNodes, 7*nNodes);
    F = zeros(nNodes, 1);

    for e = 1:nElem
        nodes = tria(e,:);
        xy = nodeCoords(nodes,:);
        x1=xy(1,1); y1=xy(1,2);
        x2=xy(2,1); y2=xy(2,2);
        x3=xy(3,1); y3=xy(3,2);

        Area = 0.5*abs((x2-x1)*(y3-y1)-(x3-x1)*(y2-y1));
        if Area < 1e-14, continue; end

        b_vec = [y2-y3; y3-y1; y1-y2];
        c_vec = [x3-x2; x1-x3; x2-x1];

        Ke = (b_vec*b_vec' + c_vec*c_vec') / (4*Area);
        fe = Area/3 * ones(3,1);

        K(nodes, nodes) = K(nodes, nodes) + Ke;
        F(nodes) = F(nodes) + fe;
    end

    tol = 1e-10;
    isBndNode = false(nNodes,1);
    for n = 1:nNodes
        xi = nodeCoords(n,1); yi = nodeCoords(n,2);
        onOuter = abs(xi+1)<tol || abs(xi-1)<tol || abs(yi+1)<tol || abs(yi-1)<tol;
        onInner = (abs(xi)<tol && yi>-tol) || (abs(yi)<tol && xi>-tol);
        if onOuter || onInner
            isBndNode(n) = true;
        end
    end

    freeDOFs = find(~isBndNode);
    u_vec = zeros(nNodes,1);
    u_vec(freeDOFs) = K(freeDOFs, freeDOFs) \ F(freeDOFs);
end
