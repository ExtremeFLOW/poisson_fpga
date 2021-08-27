program poisson_fpga
  use neko
  use cg_fpga
  implicit none

  character(len=NEKO_FNAME_LEN) :: fname, lxchar
  type(mesh_t) :: msh
  type(file_t) :: nmsh_file, mf
  type(space_t), target :: Xh
  type(coef_t), target :: c_Xh
  type(dofmap_t) :: dm_Xh
  type(gs_t) :: gs_Xh
  type(dirichlet_t) :: dir_bc
  type(ksp_monitor_t) :: ksp_results
  type(cg_fpga_t) :: solver
  type(bc_list_t) :: bclst
  type(field_t), target :: w, x, p, r
  type(ax_helm_t) :: ax
  real(kind=rp), allocatable :: f(:)
  real(kind=rp), target ::  rtz1, rtz2, beta, rnorm, rtr
  integer :: argc, lx, m, n_glb, niter, ierr
  character(len=80) :: kernel_file_name
  integer :: i, idevice, iplatform
  integer, target :: n

  argc = command_argument_count()

  if ((argc .lt. 3) .or. (argc .gt. 3)) then
     write(*,*) 'Usage: ./poisson <neko mesh> <N> <bitsteam>'
     stop
  end if
  niter = 1000

  idevice = 1
  iplatform = 2
  
  call neko_init 
  
  call get_command_argument(1, fname)
  call get_command_argument(2, lxchar)
  call get_command_argument(3, kernel_file_name)
  read(lxchar, *) lx
  
  nmsh_file = file_t(fname)
  call nmsh_file%read(msh)  

  call space_init(Xh, GLL, lx, lx, lx)

  dm_Xh = dofmap_t(msh, Xh)
  
  call gs_init(gs_Xh, dm_Xh)
  
  call coef_init(c_Xh, gs_Xh)
  
  call field_init(w, dm_Xh, "w")
  call field_init(x, dm_Xh, "x")
  call field_init(r, dm_Xh, "r")
  call field_init(p, dm_Xh, "p")

  n = Xh%lx * Xh%ly * Xh%lz * msh%nelv
  call dir_bc%init(dm_Xh)
  call dir_bc%set_g(real(0.0d0,rp))
  call set_bc(dir_bc, msh)
  call dir_bc%finalize()
  call bc_list_init(bclst)
  call bc_list_add(bclst,dir_bc)
  allocate(f(n))
  call rzero(f,n)
  call set_f(f, c_Xh%mult, dm_Xh, n, gs_Xh)
  call bc_list_apply(bclst,f,n)
  call solver%init(n)
  call cg_fpga_init_device(solver, idevice, iplatform, kernel_file_name, lx) 
 
  n_glb = Xh%lxyz * msh%glb_nelv

  call rzero(x%x, n)
  call rzero(w%x, n)
  call copy(r%x, f, n)
  rtr = glsc3(r%x, c_xh%mult, r%x, n)
  
  call Ax%compute(w%x,r%x, c_Xh, msh, Xh)
  call gs_op_vector(gs_Xh, w%x, n, GS_OP_ADD)
  call bc_list_apply_scalar(bclst, w%x, n)
  rtz1 = rtr
  beta = 0d0 
  call cg_fpga_populate(solver, r, w, x, Xh, c_Xh,bclst, gs_Xh, rtz1, beta)
  call set_timer_flop_cnt(0, msh%glb_nelv, Xh%lx, niter, n_glb)
  ksp_results = solver%solve(Ax, x, f, n, c_Xh, bclst, gs_Xh, niter)
  call set_timer_flop_cnt(1, msh%glb_nelv, Xh%lx, niter, n_glb)
  call fpga_get_data(solver, x)
  fname = 'out.fld'
  mf =  file_t(fname)
  call mf%write(x)
 
  
  call solver%free() 
  call field_free(r)
  call field_free(w)
  call field_free(x)
  call field_free(p)
  call coef_free(c_Xh)
  call gs_free(gs_Xh)
  call space_free(Xh)   
  call mesh_free(msh)

  
  call neko_finalize

end program poisson_fpga

subroutine set_timer_flop_cnt(iset, nelt, nx1, niter, n)
  use comm
  use num_types
  implicit none

  integer :: iset
  integer, intent(inout) :: nelt
  integer, intent(inout) :: nx1
  integer, intent(inout) :: niter
  integer, intent(inout) :: n
  real(kind=dp), save :: time0, time1, mflops, flop_a, flop_cg, eff_BW, arr_time
  integer :: ierr  
  real(kind=dp) :: nxyz, nx
  
  nx = dble(nx1)
  nxyz = dble(nx1 * nx1 * nx1)
  call MPI_Barrier(NEKO_COMM, ierr)  
  if (iset .eq. 0) then
     time0 = MPI_Wtime()
  else
     time1 = MPI_Wtime()
     time1 = time1-time0
     flop_a = (15d0 * nxyz + 12d0 * nx * nxyz) * dble(nelt) * dble(niter)
     flop_cg = dble(niter)*15d0*dble(nelt)*nxyz
     eff_BW = 6d0/nx/dble(nelt)**(1d0/3d0) + 3d0 + 7d0 + 10d0 + ((nx)**3-(nx-2d0)**3)/nx**3 * 2d0
     eff_BW = rp * eff_Bw * nxyz * dble(nelt) * dble(niter)/(1.d9*time1)
     arr_time= rp * nxyz * dble(nelt) * dble(niter)/(1.d9*time1)
     if (time1 .gt. 0) mflops = (flop_a + flop_cg)/(1.d6*time1)
     if (pe_rank .eq. 0) then
        write(6,*)
        write(6,1) nelt,pe_size,nx1
        write(6,2) mflops, time1
        write(6,3) eff_BW
        write(6,4) arr_time
     endif
1    format('nelt = ',i7, ', np = ', i9,', nx1 = ', i7)
2    format('Tot MFlops = ', 1pe12.4, ', Time        = ', e12.4)
3    format('Effective bandwidth, GB/s', 1pe12.4)
4    format('arraysize/time GB/s', 1pe12.4)
  endif

end subroutine set_timer_flop_cnt

subroutine set_data(u, v, n)
  use num_types
  implicit none
  
  real(kind=dp), intent(inout), dimension(n) :: u
  real(kind=dp), intent(inout), dimension(n) :: v
  integer,  intent(inout) :: n
  real(kind=dp) :: arg
  integer :: i

  do i = 1, n
     arg = (i * i)
     arg =  cos(arg)
     u(i) = sin(arg)
     v(i) = sin(arg)
  end do

end subroutine set_data
