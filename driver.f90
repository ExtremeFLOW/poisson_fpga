program axbench
  use neko
  use clfortran
  use clroutines
  use ISO_C_BINDING  
  implicit none

  character(len=NEKO_FNAME_LEN) :: fname, lxchar
  type(mesh_t) :: msh
  type(file_t) :: nmsh_file
  type(space_t), target :: Xh
  type(coef_t), target :: c_Xh
  type(dofmap_t) :: dm_Xh
  type(gs_t) :: gs_Xh
  type(dirichlet_t) :: dir_bc
  type(bc_list_t) :: bclst
  type(field_t), target :: w, x, p, r
  real(kind=rp), allocatable :: f(:)
  real(kind=rp), target ::  rtz1, rtz2, beta, rnorm, rtr
  integer :: argc, lx, m, n_glb, niter, ierr
  character(len=80) :: suffix, kernel_file_name
  integer :: i
  integer, target :: n
  integer(c_intptr_t), target :: cmd_queue, cl_bk5_kernel, cl_cg_kernel
  integer(c_intptr_t), target :: cl_x, cl_w, cl_p, cl_res
  integer(c_intptr_t), target :: cl_g1, cl_g2, cl_g3, cl_g4, cl_g5, cl_g6
  integer(c_intptr_t), target :: cl_mult, cl_dx, cl_dxt, cl_rtz1, cl_rtz2, cl_beta
  character*1 ans
  integer(c_size_t) :: byte_size, element_size, dx_size
  integer(c_int32_t) :: err
  character(len=1024) :: options, kernel_name
  character(len=1, kind=c_char),allocatable :: kernel_str(:)
  integer(c_size_t),target :: globalsize, length
  integer(c_size_t),target :: localsize(3)
  integer(c_intptr_t), target :: prog, context, binary_status
  integer(c_intptr_t), allocatable, target :: platform_ids(:), device_ids(:)
  integer :: iplatform
  integer(c_int) :: num_platforms
  integer :: idevice, irec
  integer :: filesize 
  character(len=1,kind=c_char), allocatable, target :: binary(:)
  character(len=1,kind=c_char), target :: c_kernel_name(1:1024)
  type(c_ptr), target :: psource, event
  argc = command_argument_count()

  if ((argc .lt. 3) .or. (argc .gt. 3)) then
     write(*,*) 'Usage: ./poisson <neko mesh> <N> <bitsteam>'
     stop
  end if
  
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
  print *, glsc2(f,f,n)
 
  niter = 1000

  idevice = 1
  iplatform = 2
  byte_size= int(8*n,8)
  dx_size = int(8*lx**2,8)

  globalsize=int(n,8)
  localsize=(/int(8,8),int(8,8),int(8,8)/)
 
  call create_device_context(iplatform, platform_ids,&
                             device_ids, context, cmd_queue)
  call query_platform_info(platform_ids(iplatform))
  call read_file(kernel_file_name,binary,filesize)
  length = filesize
  psource = C_LOC(binary)
  prog = clCreateProgramWithBinary(context, 1, C_LOC(device_ids(idevice)),&
                                   C_LOC(length),C_LOC(psource),C_NULL_PTR,err)
  
  err=clBuildProgram(prog, 0, C_NULL_PTR,C_NULL_PTR,C_NULL_FUNPTR,C_NULL_PTR)
  
  kernel_name = "pre_ax"
  irec=len(trim(kernel_name))
  do i=1,irec
     c_kernel_name(i)=kernel_name(i:i)
  enddo
  c_kernel_name(irec+1)=C_NULL_CHAR
  cl_bk5_kernel=clCreateKernel(prog,C_LOC(c_kernel_name),err)
  if (err.ne.0) stop 'clCreateKernel'
    
  kernel_name = "post_ax"
  irec=len(trim(kernel_name))
  do i=1,irec
     c_kernel_name(i)=kernel_name(i:i)
  enddo
  c_kernel_name(irec+1)=C_NULL_CHAR
  cl_cg_kernel=clCreateKernel(prog,C_LOC(c_kernel_name),err)
  if (err.ne.0) stop 'clCreateKernel'
    
 

  err = clReleaseProgram(prog)
  if (err.ne.0) stop 'clReleaseProgram'
  cl_x = clCreateBuffer(context,ior(CL_MEM_READ_WRITE,CL_CHANNEL_1_INTELFPGA),&
                        byte_size,C_NULL_PTR, err)
  if (err.ne.0) stop 'clCreateBuffer'
  cl_w = clCreateBuffer(context,ior(CL_MEM_READ_WRITE, CL_CHANNEL_2_INTELFPGA),&
                        byte_size,C_NULL_PTR, err)
  if (err.ne.0) stop 'clCreateBuffer'
  cl_res = clCreateBuffer(context,ior(CL_MEM_READ_WRITE,CL_CHANNEL_3_INTELFPGA),&
                        byte_size,C_NULL_PTR, err)
  if (err.ne.0) stop 'clCreateBuffer'
  cl_p = clCreateBuffer(context,ior(CL_MEM_READ_WRITE,CL_CHANNEL_4_INTELFPGA),&
                        byte_size,C_NULL_PTR, err)
  if (err.ne.0) stop 'clCreateBuffer'

  cl_mult = clCreateBuffer(context,ior(CL_MEM_READ_ONLY,CL_CHANNEL_4_INTELFPGA),&
                        byte_size,C_NULL_PTR, err)
  if (err.ne.0) stop 'clCreateBuffer'
  cl_g1 = clCreateBuffer(context,&
                         ior(CL_MEM_READ_ONLY, CL_CHANNEL_3_INTELFPGA),&
                         byte_size,C_NULL_PTR, err)
  if (err.ne.0) stop 'clCreateBuffer'
  cl_g2 = clCreateBuffer(context,&
                         ior(CL_MEM_READ_ONLY, CL_CHANNEL_4_INTELFPGA),&
                         byte_size,C_NULL_PTR, err)
  if (err.ne.0) stop 'clCreateBuffer'
  cl_g3 = clCreateBuffer(context,&
                         ior(CL_MEM_READ_ONLY, CL_CHANNEL_1_INTELFPGA),&
                         byte_size,C_NULL_PTR, err)
  if (err.ne.0) stop 'clCreateBuffer'
  cl_g4 = clCreateBuffer(context,&
                         ior(CL_MEM_READ_ONLY, CL_CHANNEL_2_INTELFPGA),&
                         byte_size,C_NULL_PTR, err)
  if (err.ne.0) stop 'clCreateBuffer'
  cl_g5 = clCreateBuffer(context,&
                         ior(CL_MEM_READ_ONLY, CL_CHANNEL_3_INTELFPGA),&
                         byte_size,C_NULL_PTR, err)
  if (err.ne.0) stop 'clCreateBuffer'
  cl_g6 = clCreateBuffer(context,& 
                         ior(CL_MEM_READ_ONLY, CL_CHANNEL_4_INTELFPGA),&
                         byte_size,C_NULL_PTR, err)
  if (err.ne.0) stop 'clCreateBuffer'
 
  cl_dx = clCreateBuffer(context,&
                         ior(CL_MEM_READ_ONLY, CL_CHANNEL_1_INTELFPGA),&
                         dx_size,C_NULL_PTR, err)
  if (err.ne.0) stop 'clCreateBuffer'
  cl_dxt = clCreateBuffer(context,&
                          ior(CL_MEM_READ_ONLY, CL_CHANNEL_2_INTELFPGA),&
                          dx_size,C_NULL_PTR, err)
  if (err.ne.0) stop 'clCreateBuffer'
  cl_rtz1 = clCreateBuffer(context,& 
                           CL_MEM_READ_WRITE,int(8,8),C_NULL_PTR, err)
  if (err.ne.0) stop 'clCreateBuffer'
  cl_rtz2 = clCreateBuffer(context,& 
                           CL_MEM_READ_WRITE,int(8,8),C_NULL_PTR, err)
  if (err.ne.0) stop 'clCreateBuffer'
  cl_beta = clCreateBuffer(context,& 
                           CL_MEM_READ_WRITE,int(8,8),C_NULL_PTR, err)
  if (err.ne.0) stop 'clCreateBuffer'
  
  err=clSetKernelArg(cl_bk5_kernel,0,sizeof(cl_res),C_LOC(cl_res))
  if (err.ne.0) stop 'clSetKernelArg'
  err=clSetKernelArg(cl_bk5_kernel,1,sizeof(cl_w),C_LOC(cl_w))
  if (err.ne.0) stop 'clSetKernelArg'
  err=clSetKernelArg(cl_bk5_kernel,2,sizeof(cl_p),C_LOC(cl_p))
  if (err.ne.0) stop 'clSetKernelArg'
  err=clSetKernelArg(cl_bk5_kernel,3,sizeof(cl_g1),C_LOC(cl_g1))
  if (err.ne.0) stop 'clSetKernelArg'
  err=clSetKernelArg(cl_bk5_kernel,4,sizeof(cl_g2),C_LOC(cl_g2))
  if (err.ne.0) stop 'clSetKernelArg'
  err=clSetKernelArg(cl_bk5_kernel,5,sizeof(cl_g3),C_LOC(cl_g3))
  if (err.ne.0) stop 'clSetKernelArg'
  err=clSetKernelArg(cl_bk5_kernel,6,sizeof(cl_g4),C_LOC(cl_g4))
  if (err.ne.0) stop 'clSetKernelArg'
  err=clSetKernelArg(cl_bk5_kernel,7,sizeof(cl_g5),C_LOC(cl_g5))
  if (err.ne.0) stop 'clSetKernelArg'
  err=clSetKernelArg(cl_bk5_kernel,8,sizeof(cl_g6),C_LOC(cl_g6))
  if (err.ne.0) stop 'clSetKernelArg'
  err=clSetKernelArg(cl_bk5_kernel,9,sizeof(cl_dx),C_LOC(cl_dx))
  if (err.ne.0) stop 'clSetKernelArg'
  err=clSetKernelArg(cl_bk5_kernel,10,sizeof(cl_dxt),C_LOC(cl_dxt))
  if (err.ne.0) stop 'clSetKernelArg'
  err=clSetKernelArg(cl_bk5_kernel,11,sizeof(cl_beta),C_LOC(cl_beta))
  if (err.ne.0) stop 'clSetKernelArg'
  err=clSetKernelArg(cl_bk5_kernel,12,sizeof(n),C_LOC(n))
  if (err.ne.0) stop 'clSetKernelArg'
  
  err=clSetKernelArg(cl_cg_kernel,0,sizeof(cl_x),C_LOC(cl_x))
  if (err.ne.0) stop 'clSetKernelArg'
  err=clSetKernelArg(cl_cg_kernel,1,sizeof(cl_p),C_LOC(cl_p))
  if (err.ne.0) stop 'clSetKernelArg'
  err=clSetKernelArg(cl_cg_kernel,2,sizeof(cl_res),C_LOC(cl_res))
  if (err.ne.0) stop 'clSetKernelArg'
  err=clSetKernelArg(cl_cg_kernel,3,sizeof(cl_w),C_LOC(cl_w))
  if (err.ne.0) stop 'clSetKernelArg'
  err=clSetKernelArg(cl_cg_kernel,4,sizeof(cl_mult),C_LOC(cl_mult))
  if (err.ne.0) stop 'clSetKernelArg'
  err=clSetKernelArg(cl_cg_kernel,5,sizeof(cl_rtz1),C_LOC(cl_rtz1))
  if (err.ne.0) stop 'clSetKernelArg'
  err=clSetKernelArg(cl_cg_kernel,6,sizeof(cl_rtz2),C_LOC(cl_rtz2))
  if (err.ne.0) stop 'clSetKernelArg'
  err=clSetKernelArg(cl_cg_kernel,7,sizeof(cl_beta),C_LOC(cl_beta))
  if (err.ne.0) stop 'clSetKernelArg'
  err=clSetKernelArg(cl_cg_kernel,8,sizeof(n),C_LOC(n))
  if (err.ne.0) stop 'clSetKernelArg'
  

 

  err = clEnqueueWriteBuffer(cmd_queue,cl_mult,CL_TRUE,0_8,byte_size,&
                             C_LOC(c_Xh%mult), 0,C_NULL_PTR,C_NULL_PTR)
  if (err .ne. 0) stop 'clEnqueueWriteBuffer'
  err = clEnqueueWriteBuffer(cmd_queue,cl_g1,CL_TRUE,0_8,byte_size,&
                             C_LOC(c_Xh%g1), 0,C_NULL_PTR,C_NULL_PTR)
  if (err .ne. 0) stop 'clEnqueueWriteBuffer'
  err = clEnqueueWriteBuffer(cmd_queue,cl_g2,CL_TRUE,0_8,byte_size,&
                             C_LOC(c_Xh%g2), 0,C_NULL_PTR,C_NULL_PTR)
  if (err .ne. 0) stop 'clEnqueueWriteBuffer'
  err = clEnqueueWriteBuffer(cmd_queue,cl_g3,CL_TRUE,0_8,byte_size,&
                             C_LOC(c_Xh%g3), 0,C_NULL_PTR,C_NULL_PTR)
  if (err .ne. 0) stop 'clEnqueueWriteBuffer'
  
  err = clEnqueueWriteBuffer(cmd_queue,cl_g4,CL_TRUE,0_8,byte_size,&
                             C_LOC(c_Xh%g4), 0,C_NULL_PTR,C_NULL_PTR)
  if (err .ne. 0) stop 'clEnqueueWriteBuffer'
  err = clEnqueueWriteBuffer(cmd_queue, cl_g5, CL_TRUE, 0_dp, byte_size,&
                             C_LOC(c_Xh%g5), 0, C_NULL_PTR, C_NULL_PTR)
  if (err .ne. 0) stop 'clEnqueueWriteBuffer'
  err = clEnqueueWriteBuffer(cmd_queue,cl_g6,CL_TRUE,0_8,byte_size,&
                             C_LOC(c_Xh%g6), 0,C_NULL_PTR,C_NULL_PTR)
  if (err.ne.0) stop 'clEnqueueWriteBuffer'
  
  err = clEnqueueWriteBuffer(cmd_queue,cl_dx,CL_TRUE,0_8,dx_size,&
                             C_LOC(Xh%dx), 0,C_NULL_PTR,C_NULL_PTR)
  if (err.ne.0) stop 'clEnqueueWriteBuffer'
  err = clEnqueueWriteBuffer(cmd_queue,cl_dxt,CL_TRUE,0_8,dx_size,&
                             C_LOC(Xh%dxt), 0,C_NULL_PTR,C_NULL_PTR)
  if (err.ne.0) stop 'clEnqueueWriteBuffer'

  n_glb = Xh%lxyz * msh%glb_nelv
  call rzero(x%x, n)
  call rzero(p%x, n)
  call rzero(w%x, n)
  call copy(r%x, f, n)
  rtr = glsc3(r%x, c_xh%mult, r%x, n)
  print *, rtr
  rtz1 = rtr
  beta = 0d0 
  !Write to the fields 
  err = clEnqueueWriteBuffer(cmd_queue,cl_w,CL_TRUE,0_8,byte_size,&
                             C_LOC(w%x), 0,C_NULL_PTR,C_NULL_PTR)
  if (err.ne.0) stop 'clEnqueueWriteBuffer'
  err = clEnqueueWriteBuffer(cmd_queue,cl_x,CL_TRUE,0_8,byte_size,&
                             C_LOC(x%x), 0,C_NULL_PTR,C_NULL_PTR)
  if (err.ne.0) stop 'clEnqueueWriteBuffer'
  err = clEnqueueWriteBuffer(cmd_queue,cl_res,CL_TRUE,0_8,byte_size,&
                             C_LOC(r%x), 0,C_NULL_PTR,C_NULL_PTR)
  if (err.ne.0) stop 'clEnqueueWriteBuffer'
  err = clEnqueueWriteBuffer(cmd_queue,cl_p,CL_TRUE,0_8,byte_size,&
                             C_LOC(p%x), 0,C_NULL_PTR,C_NULL_PTR)
  if (err.ne.0) stop 'clEnqueueWriteBuffer'
  err = clEnqueueWriteBuffer(cmd_queue,cl_beta,CL_TRUE,0_8,int(8,8),&
                             C_LOC(beta), 0,C_NULL_PTR,C_NULL_PTR)
  if (err.ne.0) stop 'clEnqueueWriteBuffer'
  err = clEnqueueWriteBuffer(cmd_queue,cl_rtz1,CL_TRUE,0_8,int(8,8),&
                             C_LOC(rtz1), 0,C_NULL_PTR,C_NULL_PTR)
  if (err.ne.0) stop 'clEnqueueWriteBuffer'
     err=clFinish(cmd_queue)
     if (err.ne.0) stop 'clFinish'
  
  call set_timer_flop_cnt(0, msh%glb_nelv, Xh%lx, niter, n_glb)
  do i = 1, niter
     err=clEnqueueTask(cmd_queue,cl_bk5_kernel,0,C_NULL_PTR,C_NULL_PTR)
     if (err .ne. 0) stop 'clEnqueueEnqueueTask bk5'
     err = clEnqueueReadBuffer(cmd_queue,cl_w,CL_TRUE,&
                               0_8,byte_size,C_LOC(w%x),0,C_NULL_PTR,C_NULL_PTR)
     if (err.ne.0) stop 'clEnqueueReadBuffer'
     err=clFinish(cmd_queue)
     if (err.ne.0) stop 'clFinish'
     call gs_op_vector(gs_Xh, w%x, n, GS_OP_ADD)
     call bc_list_apply_scalar(bclst, w%x, n)
     err=clEnqueueWriteBuffer(cmd_queue,cl_w,CL_TRUE,0_8,&
                              byte_size,C_LOC(w%x), 0,C_NULL_PTR,C_NULL_PTR)
     if (err.ne.0) stop 'clEnqueueWriteBuffer bk5'
  
     err=clEnqueueTask(cmd_queue,cl_cg_kernel,0,C_NULL_PTR,C_NULL_PTR)
     if (err .ne. 0) stop 'clEnqueueEnqueueTask cg'
     err = clEnqueueReadBuffer(cmd_queue,cl_rtz1,CL_TRUE,&
                               0_8,int(8,8),C_LOC(rtz1),0,C_NULL_PTR,C_NULL_PTR)
     if (err.ne.0) stop 'clEnqueueReadBuffer'
     err=clFinish(cmd_queue)
     rnorm = sqrt(rtz1)
     print *, rnorm, rtz1
  end do

  err=clFinish(cmd_queue)
  if (err.ne.0) stop 'clFinish'
  err = clReleaseCommandQueue(cmd_queue) 
  err = clReleaseContext(context)  
  err = clReleaseMemObject(cl_x)
  if (err.ne.0) stop 'clRelasseMemObj'
  err = clReleaseMemObject(cl_w)
  if (err.ne.0) stop 'clRelasseMemObj'
  err = clReleaseMemObject(cl_g1)
  if (err.ne.0) stop 'clRelasseMemObj'
  err = clReleaseMemObject(cl_g2)
  if (err.ne.0) stop 'clRelasseMemObj'
  err = clReleaseMemObject(cl_g3)
  if (err.ne.0) stop 'clRelasseMemObj'
  err = clReleaseMemObject(cl_g4)
  if (err.ne.0) stop 'clRelasseMemObj'
  err = clReleaseMemObject(cl_g5)
  if (err.ne.0) stop 'clRelasseMemObj'
  err = clReleaseMemObject(cl_g6)
  if (err.ne.0) stop 'clRelasseMemObj'

  call set_timer_flop_cnt(1, msh%glb_nelv, Xh%lx, niter, n_glb)
  
  call field_free(r)
  call field_free(w)
  call field_free(x)
  call field_free(p)
  call coef_free(c_Xh)
  call gs_free(gs_Xh)
  call space_free(Xh)   
  call mesh_free(msh)

  
  call neko_finalize

end program axbench

subroutine set_timer_flop_cnt(iset, nelt, nx1, niter, n)
  use comm
  use num_types
  implicit none

  integer :: iset
  integer, intent(inout) :: nelt
  integer, intent(inout) :: nx1
  integer, intent(inout) :: niter
  integer, intent(inout) :: n
  real(kind=dp), save :: time0, time1, mflops, flop_a, flop_cg
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
     if (time1 .gt. 0) mflops = (flop_a + flop_cg)/(1.d6*time1)
     if (pe_rank .eq. 0) then
        write(6,*)
        write(6,1) nelt,pe_size,nx1
        write(6,2) mflops, time1
     endif
1    format('nelt = ',i7, ', np = ', i9,', nx1 = ', i7)
2    format('Tot MFlops = ', 1pe12.4, ', Time        = ', e12.4)
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
