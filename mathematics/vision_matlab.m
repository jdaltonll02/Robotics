% MATLAB script for camera model and synthetic data visualization
% This script demonstrates pinhole camera projection and synthetic depth map generation

% Camera intrinsics
fx = 320; fy = 320; cx = 160; cy = 120;
K = [fx 0 cx; 0 fy cy; 0 0 1];

% Generate synthetic 3D points (e.g., a plane)
[X, Y] = meshgrid(-1:0.05:1, -1:0.05:1);
Z = 2 + 0.5*X + 0.2*Y;
pts3D = [X(:) Y(:) Z(:)]';

% Project to image
pts2D = K * pts3D;
pts2D = pts2D ./ pts2D(3,:);

% Visualize
figure;
scatter(pts2D(1,:), pts2D(2,:), 10, Z(:), 'filled');
colorbar; title('Projected Depth Map'); xlabel('u'); ylabel('v');
