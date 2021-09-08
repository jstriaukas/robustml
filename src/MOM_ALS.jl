using LinearAlgebra
using Statistics
using Random

function abs_val(d)
	return abs(d)
end

function obj(X, y, w, lam, tau)
    n, ncol = size(X);
    pen = 0;
    obj_fun = 0;
    r = y - X * w;
    rw = zeros(n);
    for q_i in 1:n
        if r[q_i] < 0
            rw[q_i] = r[q_i] * r[q_i] * (1 - tau);
        else
            rw[q_i] = r[q_i] * r[q_i] * tau;
        end
    end
    pen = lam * sum(broadcast(abs, w));
    obj_fun = sum(rw) + pen;
    return obj_fun;
end

function median_index(vector)
    med = median(vector);
    abs_demed_vector = broadcast(abs, vector .- med);
    idx = argmin(abs_demed_vector);
    return idx;
end

function buckets(K_buckets, nrow)
    tmp_buckets_id = range(0, stop=K_buckets-1, length=K_buckets);
    tmp2buckets_id = repeat(tmp_buckets_id, 1, Int(ceil(nrow/K_buckets)));
    buckets_id = sort(vec(tmp2buckets_id)[1:nrow]);
    return buckets_id;
end

function cvfolds(nfolds, nrow)
    tmp_foldsid = range(0, stop=nfolds-1, length=nfolds);
    tmp2_foldsid = repeat(tmp_foldsid, 1, Int(ceil(nrow/nfolds)));
    foldsid = sort(vec(tmp2_foldsid)[1:nrow]);
    return foldsid;
end

function block_X(X, which_idx)
    return X[which_idx,:];
end

function block_y(y, which_idx)
    return y[which_idx];
end

function mom_obj(X, y, w, w_prime, lam, tau) 
    loss = obj(X, y, w, lam, tau);
    loss_prime = obj(X, y, w_prime, lam, tau);
    loss_mom = loss - loss_prime;
    return loss_mom;
end

function f_grad(X, y, w, tau)
    n, ncol = size(X);
    r = y - X * w;
    grad_tmp = zeros(n);
    for q_i in 1:n
        if r[q_i] < 0
            grad_tmp[q_i] = 2 * r[q_i] * (1 - tau);
        else
            grad_tmp[q_i] = 2 * r[q_i] * tau;
        end
    end
    grad = -transpose(X) * grad_tmp;
    return grad;
end

function block(X, y, w, w_prime, K, tau, is_random)
	vect_means = zeros(K);
	n = size(y)[1];
	buckets_id = buckets(K, n);
	if is_random == true
		buckets_id = shuffle(buckets_id);
	end
	for k in 1:K
		Xk = X[findall(x -> x==(k-1), buckets_id), :];
		yk = y[findall(x -> x==(k-1), buckets_id)];
		excess_loss_k = obj(Xk, yk, w, 0, tau) - obj(Xk, yk, w_prime, 0, tau);
		vect_means[k] = excess_loss_k;
	end
	idx = median_index(vect_means);
	which_idx = findall(x -> x==(idx-1), buckets_id);
	return which_idx;
end


function block_cv(X_test, y_test, w, K_prime, tau)
	vect_means = zeros(K_prime);
	n = size(y_test)[1];
	buckets_id = buckets(K_prime, n);
	for k in 1:K_prime
		Xk = X_test[findall(x -> x==(k-1), buckets_id), :];
		yk = y_test[findall(x -> x==(k-1), buckets_id)];
		excess_loss_k = obj(Xk, yk, w, 0, tau);	
		vect_means[k] = excess_loss_k;
	end
  idx = median_index(vect_means);
  which_idx = findall(x -> x==(idx-1), buckets_id);
  return which_idx;
end

function soft_threshold(w, mu)
	p = size(w)[1];
	max_tmp = zeros(p);
	for i in 1:p
		if abs_val(w[i]) > mu
			max_tmp[i] = abs_val(w[i]) - mu;
		else
			max_tmp[i] = 0.0;
		end
	end  
	st = sign.(w) .* max_tmp;
	return st;
end

function pg_als(X, y, beta_0, tau, max_iter, lam) 
	w = beta_0;
	w_prev = w;
	obj_pg_als = zeros(max_iter+1);
	tmp = obj(X, y, w, lam, tau);
	obj_pg_als[1] = tmp;
  
	beta = 1.2;
	delta = 1.0;
	LL = 1.0;
	for j in 1:max_iter
		w_prev = w;
		delta = 1.0;
		obj_val = obj(X, y, w, lam, tau); 
		while (delta > 0.001)
			gamma = 1.0 / LL;
			w = w_prev - gamma * f_grad(X, y, w_prev, tau);
			w = soft_threshold(w, lam * gamma);
			obj_val_new = obj(X, y, w, lam, tau);
			f_grad_w = sum(transpose(f_grad(X, y, w_prev, tau)) * (w - w_prev));
			delta = obj_val_new - obj_val - f_grad_w - (LL / 2) * norm(w-w_prev, 2) ^ 2;
			LL = LL * beta;
		end
		LL = LL / beta;
		tmp = obj(X, y, w, lam, tau);
		obj_pg_als[j+1] = tmp;
	end
	out = Dict("fitted_values" => obj_pg_als, "beta" => w);
	return out;
end


function pg_mom_als(X, y, beta_0, tau, K, max_iter, lam, epsil, is_random) 
	w = beta_0;
	w_prime = beta_0;
	mom_obj_pg_als = zeros(max_iter+1);
	tmp = mom_obj(X, y, w, w_prime, lam, tau);
	mom_obj_pg_als[1] = tmp;
	beta = 1.2;
	stopping = false;
	counter = 0.0;
	tmp_obj = 0.0;
	tmp_obj_prev = 0.0;
	while (counter < max_iter && stopping == false)
		counter = counter + 1.0;
		LL = 1.0; 
		w_prev = w;
		delta = ones(1);
		idx_k = block(X, y, w, w_prime, K, tau, is_random);
		Xk = block_X(X, idx_k);
		yk = block_y(y, idx_k);
		obj_val = obj(Xk, yk, w, lam, tau); 
		while (delta[1] > 0.001)
			gamma = 1 / LL;
			w = w_prev - gamma * f_grad(Xk, yk, w_prev, tau);
			w = soft_threshold(w, lam * gamma);
			delta = obj(Xk, yk, w, lam, tau) - obj_val - transpose(Array(f_grad(Xk, yk, w_prev, tau))) * (w - w_prev) - (LL / 2) * norm(w-w_prev, 2) ^ 2;
			LL = LL * beta;
		end
		LL_prime = 1.0;
		w_prime_prev = w_prime;
		delta_prime = ones(1);
		idx_k = block(X, y, w, w_prime, K, tau, is_random);
		Xk = block_X(X, idx_k);
		yk = block_y(y, idx_k);
		obj_val = obj(Xk, yk, w_prime, lam, tau); 
		while (delta_prime[1] > 0.001)
			gamma = 1 / LL_prime;
			w_prime = w_prime_prev - gamma * f_grad(Xk, yk, w_prime_prev, tau);
			w_prime = soft_threshold(w_prime, lam * gamma);
			delta_prime = obj(Xk, yk, w_prime, lam, tau) - obj_val - transpose(Array(f_grad(Xk, yk, w_prime_prev, tau))) * (w_prime - w_prime_prev) - (LL_prime / 2) * norm(w_prime - w_prime_prev, 2) ^ 2;
			LL_prime = LL_prime * beta;
		end
		tmp_obj_prev = tmp_obj;
		tmp_obj = mom_obj(X, y, w, w_prime, lam, tau);
		mom_obj_pg_als[Int(counter)+1] = tmp_obj;
		#if (abs_val(tmp_obj) <= epsil)
		#	if (abs_val(tmp_obj) + abs_val(tmp_obj_prev) <= 2 * epsil)
		#		stopping = true;
		#		counter = counter + 1.0;
		#	end
		#end
	end
	out = Dict("fitted_values" => mom_obj_pg_als, "beta" => w, "beta_prime" => w_prime, "counter" => counter, "stopping" => stopping);
	return out
end


function pg_mom_als_w(X, y, beta_0, tau, K, max_iter, lam, epsil, is_random)
	w = beta_0;
	w_prime = beta_0;
	mom_obj_pg_als = zeros(max_iter+1);
	tmp = mom_obj(X, y, w, w_prime, lam, tau);
	mom_obj_pg_als[1] = tmp;
	beta = 1.2;
    stopping = false;
	counter = 0.0;
	tmp_obj = 0.0;
	tmp_obj_prev = 0.0;
	while (counter < max_iter && stopping == false) 
		counter = counter + 1.0;
		LL = 1.0; 
		w_prev = w;
		delta = ones(1);
		idx_k = block(X, y, w, w_prime, K, tau, is_random);
		Xk = block_X(X, idx_k);
		yk = block_y(y, idx_k);
		obj_val = obj(Xk, yk, w, lam, tau); 
		while (delta[1] > 0.001)
			gamma = 1 / LL;
			w = w_prev - gamma * f_grad(Xk, yk, w_prev, tau);
			w = soft_threshold(w, lam * gamma);
			delta = obj(Xk, yk, w, lam, tau) - obj_val - transpose(Array(f_grad(Xk, yk, w_prev, tau)))*(w - w_prev)-(LL/2)* norm(w-w_prev, 2) ^ 2;
			LL = LL*beta;
		end
		LL_prime = 1.0;
		w_prime_prev = w_prime;
		delta_prime = ones(1);
		idx_k = block(X, y, w, w_prime, K, tau, is_random);
		Xk = block_X(X, idx_k);
		yk = block_y(y, idx_k);
		obj_val = obj(Xk, yk, w_prime, lam, tau); 
		while (delta_prime[1] > 0.001)
			gamma = 1 / LL_prime;
			w_prime = w_prime_prev - gamma * f_grad(Xk, yk, w_prime_prev, tau);
			w_prime = soft_threshold(w_prime, lam * gamma);
			delta_prime = obj(Xk, yk, w_prime, lam, tau) - obj_val - transpose(Array(f_grad(Xk, yk, w_prime_prev, tau)))*(w_prime - w_prime_prev)-(LL_prime/2)*norm(w_prime - w_prime_prev, 2) ^ 2;
			LL_prime = LL_prime*beta;
		end
		tmp_obj_prev = tmp_obj;
		tmp_obj = mom_obj(X, y, w, w_prime, lam, tau);
		mom_obj_pg_als[Int(counter)+1] = tmp_obj;
		if (abs_val(tmp_obj) <= epsil)
			if (abs_val(tmp_obj)+abs_val(tmp_obj_prev) <= 2*epsil)
				stopping = true;
				counter = counter + 1.0;
			end
		end
	end
	return w;
end


"""

X = [1 2 3; 4 5 6; 7 8 9];
y = [2, -4, 6];
w = [1, 2, 5];
w_prime = [1, 2, 3];
tau = 0.5;
lam = 2.0;
K = 3;
K_buckets = 3;
nrow = 10;
max_iter = 10;
epsil = 0.001;
is_random = false;

println(obj(X, y, w, lam, tau))
println(median_index(y));
println(buckets(K_buckets, nrow));
println(cvfolds(K_buckets, nrow));
println(block_X(X, [1,2]));
println(block_y(X, [1,2]));
println(mom_obj(X, y, w, w_prime, lam, tau));
println(f_grad(X, y, w, tau));
println(block(X, y, w, w_prime, K, tau, is_random))
println(pg_als(X, y, w, tau, max_iter, lam))
println(pg_mom_als(X, y, w, tau, K, max_iter, lam, epsil, is_random))
println(pg_mom_als_w(X, y, w, tau, K, max_iter, lam, epsil, is_random))

"""