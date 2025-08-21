import torch







def _deterministic_inner_product(fs: torch.tensor,
                                 gs: torch.tensor, ) -> torch.tensor:
    """Approximates the L2 inner product between fs and gs, in a Hilbert space ℋ = {F : X → ℝᵐ},
    using a Monte Carlo approximation.
    The inner product is defined as: ⟨f, g⟩ := 1/V ∫ f(x)ᵀ g(x) dx ≈ 1/n Σᵢ f(xᵢ)ᵀ g(xᵢ)
    Note we are scaling the L2 inner product by 1/volume, which removes volume from the monte carlo approximation.
    Since scaling an inner product is still a valid inner product, this is still an inner product.

    Args:
    fs: torch.tensor: The first set of function outputs. Shape (n_functions, n_datapoints, input_size, n_basis1)
    gs: torch.tensor: The second set of function outputs. Shape (n_functions, n_datapoints, input_size, n_basis2)

    Returns:
    torch.tensor: The inner product between fs and gs. Shape (n_functions, n_basis1, n_basis2)
    """

    assert len(fs.shape) == 4, f"Expected fs to have shape (f,d,m,k), got {fs.shape}"
    assert len(gs.shape) == 4, f"Expected gs to have shape (f,d,m,l), got {gs.shape}"
    assert fs.shape[0] == gs.shape[0], f"Expected fs and gs to have the same number of functions, got {fs.shape[0]} and {gs.shape[0]}"
    assert fs.shape[1] == gs.shape[1], f"Expected fs and gs to have the same number of datapoints, got {fs.shape[1]} and {gs.shape[1]}"
    assert fs.shape[2] == gs.shape[2], f"Expected fs and gs to have the same output size, got {fs.shape[2]} and {gs.shape[2]}"

    # compute inner products via MC integration
    element_wise_inner_products = torch.einsum("fdmk,fdml->fdkl", fs, gs)
    inner_product = torch.mean(element_wise_inner_products, dim=1)
    return inner_product


def _pdf_inner_product(fs: torch.tensor,
                       gs: torch.tensor, ) -> torch.tensor:
    """ Approximates the logit version of the inner product between probability density functions f: X -> p_x(X).
    ⟨f, g⟩ = ∫ₓ (f(x) − f̄(x))(g(x) − ḡ(x)) dx ≈ 1/n Σᵢ₌₁ⁿ (f(xᵢ) − f̄(xᵢ))(g(xᵢ) − ḡ(xᵢ))

    Args:
    fs: torch.tensor: The first set of function outputs. Shape (n_functions, n_datapoints, input_size, n_basis1)
    gs: torch.tensor: The second set of function outputs. Shape (n_functions, n_datapoints, input_size, n_basis2)

    Returns:
    torch.tensor: The inner product between fs and gs. Shape (n_functions, n_basis1, n_basis2)
    """

    assert len(fs.shape) == 4, f"Expected fs to have shape (f,d,m,k), got {fs.shape}"
    assert len(gs.shape) == 4, f"Expected gs to have shape (f,d,m,l), got {gs.shape}"
    assert fs.shape[0] == gs.shape[0], f"Expected fs and gs to have the same number of functions, got {fs.shape[0]} and {gs.shape[0]}"
    assert fs.shape[1] == gs.shape[1], f"Expected fs and gs to have the same number of datapoints, got {fs.shape[1]} and {gs.shape[1]}"
    assert fs.shape[2] == gs.shape[2] == 1, f"Expected fs and gs to have the same output size, which is 1 for the stochastic case since it learns the pdf(x), got {fs.shape[2]} and {gs.shape[2]}"


    # compute means and subtract them
    mean_f = torch.mean(fs, dim=1, keepdim=True)
    mean_g = torch.mean(gs, dim=1, keepdim=True)
    fs = fs - mean_f
    gs = gs - mean_g

    # compute inner products
    element_wise_inner_products = torch.einsum("fdmk,fdml->fdkl", fs, gs)
    inner_product = torch.mean(element_wise_inner_products, dim=1)

    return inner_product


def _categorical_inner_product(fs: torch.tensor,
                               gs: torch.tensor, ) -> torch.tensor:
    """ Approximates the inner product between discrete conditional probability distributions.
        f : X → Δᵐ

    Args:
    fs: torch.tensor: The first set of function outputs. Shape (n_functions, n_datapoints, n_categories, n_basis1)
    gs: torch.tensor: The second set of function outputs. Shape (n_functions, n_datapoints, n_categories, n_basis2)

    Returns:
    torch.tensor: The inner product between fs and gs. Shape (n_functions, n_basis1, n_basis2)
    """

    assert len(fs.shape) == 4, f"Expected fs to have shape  (f,d,m,k), got {fs.shape}"
    assert len(gs.shape) == 4, f"Expected gs to have shape  (f,d,m,l), got {gs.shape}"
    assert fs.shape[0] == gs.shape[ 0], f"Expected fs and gs to have the same number of functions, got {fs.shape[0]} and {gs.shape[0]}"
    assert fs.shape[1] == gs.shape[1], f"Expected fs and gs to have the same number of datapoints, got {fs.shape[1]} and {gs.shape[1]}"
    assert fs.shape[2] == gs.shape[2], f"Expected fs and gs to have the same output size, which is the number of categories in this case, got {fs.shape[2]} and {gs.shape[2]}"

    # compute means and subtract them
    mean_f = torch.mean(fs, dim=2, keepdim=True)
    mean_g = torch.mean(gs, dim=2, keepdim=True)
    fs = fs - mean_f
    gs = gs - mean_g

    # compute inner products
    element_wise_inner_products = torch.einsum("fdmk,fdml->fdkl", fs, gs)
    inner_product = torch.mean(element_wise_inner_products, dim=1)

    return inner_product


# If you are implementing a new inner product, first import Function encoder, then append a new type:
# e.g. INNER_PRODUCTS["new_type"] = _new_inner_product_function
INNER_PRODUCTS = {
    "deterministic": _deterministic_inner_product,
    "pdf": _pdf_inner_product,
    "categorical": _categorical_inner_product,
}
