!> Defines various Conjugate Gradient methods
module cg_banked_fpga
  use krylov
  use math
  use num_types
  use clroutines
  use iso_c_binding
  implicit none

  !> FPGA unpreconditioned conjugate gradient method
  type, public, extends(ksp_t) :: cg_banked_fpga_t
     real(kind=rp), allocatable :: w(:)
     real(kind=rp), allocatable :: r(:)
     real(kind=rp), allocatable :: p(:)
     integer(c_intptr_t) :: cmd_queue, cl_cg_kernel, context
     integer(c_intptr_t), dimension(4) :: cl_x, cl_w, cl_p, cl_res, cl_mult
     integer(c_intptr_t), dimension(4) :: cl_g1, cl_g2, cl_g3, cl_g4, cl_g5, cl_g6
     integer(c_intptr_t) :: cl_dx, cl_dxt, cl_rtz1, cl_rtz2, cl_beta, cl_mask
     integer(c_intptr_t) :: cl_b, cl_dg, cl_gd, cl_v 
     integer(c_size_t) :: real_size = rp 
     integer :: nbanks = 4
     integer(c_size_t) :: array_size, element_size, dx_size
     integer :: n, lo, nb, gs_m
 
   contains
     procedure, pass(this) :: init => cg_banked_fpga_init
     procedure, pass(this) :: free => cg_banked_fpga_free
     procedure, pass(this) :: solve => cg_banked_fpga_solve
  end type cg_banked_fpga_t

contains

  !> Initialise a standard PCG solver
  subroutine cg_banked_fpga_init(this, n, M, rel_tol, abs_tol)
    class(cg_banked_fpga_t), intent(inout) :: this
    class(pc_t), optional, intent(inout), target :: M
    integer, intent(in) :: n
    real(kind=rp), optional, intent(inout) :: rel_tol
    real(kind=rp), optional, intent(inout) :: abs_tol

        
    call this%free()
    
    allocate(this%w(n))
    allocate(this%r(n))
    allocate(this%p(n))

    if (present(rel_tol) .and. present(abs_tol)) then
       call this%ksp_init(rel_tol, abs_tol)
    else if (present(rel_tol)) then
       call this%ksp_init(rel_tol=rel_tol)
    else if (present(abs_tol)) then
       call this%ksp_init(abs_tol=abs_tol)
    else
       call this%ksp_init()
    end if
    !> Size of arrays in bytes
    this%array_size= rp*n
    this%n = n
          
  end subroutine cg_banked_fpga_init

  subroutine cg_banked_fpga_init_device(this,idevice, iplatform, bitstream, lx)
    class(cg_banked_fpga_t), target :: this
    integer :: idevice, iplatform
    character(len=80) :: bitstream
    integer(c_int32_t) :: err
    character(len=1024) :: kernel_name
    integer(c_size_t),target ::  length
    integer(c_intptr_t), target :: prog
    integer(c_intptr_t), allocatable, target :: platform_ids(:), device_ids(:)
    integer :: irec, i, k
    integer :: filesize , lx
    character(len=1,kind=c_char), allocatable, target :: binary(:)
    character(len=1,kind=c_char), target :: c_kernel_name(1:1024)
    type(c_ptr), target :: psource  

    this%dx_size = rp*lx**2
    call create_device_context(iplatform, platform_ids,&
                               idevice, device_ids, this%context, this%cmd_queue)
    call query_platform_info(platform_ids(iplatform))
    call read_file(bitstream,binary,filesize)
    length = filesize
    psource = C_LOC(binary)
    prog = clCreateProgramWithBinary(this%context, 1, C_LOC(device_ids(idevice)),&
                                     C_LOC(length),C_LOC(psource),C_NULL_PTR,err)
    if (err .ne. 0) stop ' prog binary'
    
    err=clBuildProgram(prog, 0, C_NULL_PTR,C_NULL_PTR,C_NULL_FUNPTR,C_NULL_PTR)
    if (err .ne. 0) stop 'not created prog'
    
    kernel_name = "cg"
    irec=len(trim(kernel_name))
    do i=1,irec
       c_kernel_name(i)=kernel_name(i:i)
    enddo
    c_kernel_name(irec+1)=C_NULL_CHAR
    this%cl_cg_kernel=clCreateKernel(prog,C_LOC(c_kernel_name),err)
    if (err.ne.0) stop 'clCreateKernel'
      
 

    err = clReleaseProgram(prog)
    if (err.ne.0) stop 'clReleaseProgram'

    call cl_create_banked_buffer(this%context, this%array_size, CL_MEM_READ_WRITE, this%cl_x)
    call cl_create_banked_buffer(this%context, this%array_size, CL_MEM_READ_WRITE, this%cl_w)
    call cl_create_banked_buffer(this%context, this%array_size, CL_MEM_READ_WRITE, this%cl_p)
    call cl_create_banked_buffer(this%context, this%array_size, CL_MEM_READ_WRITE, this%cl_res)
    call cl_create_banked_buffer(this%context, this%array_size, CL_MEM_READ_ONLY, this%cl_mult)
    call cl_create_banked_buffer(this%context, this%array_size, CL_MEM_READ_ONLY, this%cl_g1)
    call cl_create_banked_buffer(this%context, this%array_size, CL_MEM_READ_ONLY, this%cl_g2)
    call cl_create_banked_buffer(this%context, this%array_size, CL_MEM_READ_ONLY, this%cl_g3)
    call cl_create_banked_buffer(this%context, this%array_size, CL_MEM_READ_ONLY, this%cl_g4)
    call cl_create_banked_buffer(this%context, this%array_size, CL_MEM_READ_ONLY, this%cl_g5)
    call cl_create_banked_buffer(this%context, this%array_size, CL_MEM_READ_ONLY, this%cl_g6)

    this%cl_dx = clCreateBuffer(this%context,&
                           ior(CL_MEM_READ_ONLY, CL_CHANNEL_1_INTELFPGA),&
                           this%dx_size,C_NULL_PTR, err)
    if (err.ne.0) stop 'clCreateBuffer'
    this%cl_dxt = clCreateBuffer(this%context,&
                            ior(CL_MEM_READ_ONLY, CL_CHANNEL_2_INTELFPGA),&
                            this%dx_size,C_NULL_PTR, err)
    if (err.ne.0) stop 'clCreateBuffer'
    this%cl_rtz1 = clCreateBuffer(this%context,& 
                             CL_MEM_READ_WRITE,int(8,8),C_NULL_PTR, err)
    if (err.ne.0) stop 'clCreateBuffer'
    this%cl_rtz2 = clCreateBuffer(this%context,& 
                             CL_MEM_READ_WRITE,int(8,8),C_NULL_PTR, err)
    if (err.ne.0) stop 'clCreateBuffer'
    this%cl_beta = clCreateBuffer(this%context,& 
                             CL_MEM_READ_WRITE,int(8,8),C_NULL_PTR, err)
    if (err.ne.0) stop 'clCreateBuffer'
    this%cl_mask = clCreateBuffer(this%context,& 
                             CL_MEM_READ_WRITE,this%array_size,C_NULL_PTR, err)
    if (err.ne.0) stop 'clCreateBuffer'
    this%cl_b = clCreateBuffer(this%context,ior(CL_MEM_READ_WRITE,CL_CHANNEL_1_INTELFPGA),&
                          this%array_size,C_NULL_PTR, err)
    if (err.ne.0) stop 'clCreateBuffer'
    this%cl_gd = clCreateBuffer(this%context,ior(CL_MEM_READ_WRITE,CL_CHANNEL_1_INTELFPGA),&
                          this%array_size,C_NULL_PTR, err)
    if (err.ne.0) stop 'clCreateBuffer'
    this%cl_dg = clCreateBuffer(this%context,ior(CL_MEM_READ_WRITE,CL_CHANNEL_1_INTELFPGA),&
                          this%array_size,C_NULL_PTR, err)
    if (err.ne.0) stop 'clCreateBuffer'
    this%cl_v = clCreateBuffer(this%context,ior(CL_MEM_READ_WRITE,CL_CHANNEL_1_INTELFPGA),&
                          this%array_size,C_NULL_PTR, err)
    if (err.ne.0) stop 'clCreateBuffer'
    do i = 1,this%nbanks 
       call cl_set_kernel_arg(this%cl_cg_kernel,this%cl_x(i), i-1)
    end do
    k = this%nbanks + 1
    do i = k,(k+this%nbanks)
       call cl_set_kernel_arg(this%cl_cg_kernel,this%cl_p(i-k+1), i-1) 
    end do
    k = k + this%nbanks
    do i = k,(k+this%nbanks)
       call cl_set_kernel_arg(this%cl_cg_kernel,this%cl_res(i-k+1), i-1) 
    end do
    k = k + this%nbanks
    do i = k,(k+this%nbanks)
       call cl_set_kernel_arg(this%cl_cg_kernel,this%cl_w(i-k+1), i-1) 
    end do
    k = k + this%nbanks
    do i = k,(k+this%nbanks)
       call cl_set_kernel_arg(this%cl_cg_kernel,this%cl_mult(i-k+1), i-1) 
    end do
    k = k + this%nbanks
    do i = k,(k+this%nbanks)
       call cl_set_kernel_arg(this%cl_cg_kernel,this%cl_g1(i-k+1), i-1) 
    end do
    k = k + this%nbanks
    do i = k,(k+this%nbanks)
       call cl_set_kernel_arg(this%cl_cg_kernel,this%cl_g2(i-k+1), i-1) 
    end do
    k = k + this%nbanks
    do i = k,(k+this%nbanks)
       call cl_set_kernel_arg(this%cl_cg_kernel,this%cl_g3(i-k+1), i-1) 
    end do
    k = k + this%nbanks
    do i = k,(k+this%nbanks)
       call cl_set_kernel_arg(this%cl_cg_kernel,this%cl_g4(i-k+1), i-1) 
    end do
    k = k + this%nbanks
    do i = k,(k+this%nbanks)
       call cl_set_kernel_arg(this%cl_cg_kernel,this%cl_g5(i-k+1), i-1) 
    end do
    k = k + this%nbanks
    do i = k,(k+this%nbanks)
       call cl_set_kernel_arg(this%cl_cg_kernel,this%cl_g6(i-k+1), i-1) 
    end do
    k = k + this%nbanks
    
    err=clSetKernelArg(this%cl_cg_kernel,k-1,sizeof(this%cl_dx),C_LOC(this%cl_dx))
    if (err.ne.0) stop 'clSetKernelArg'
    err=clSetKernelArg(this%cl_cg_kernel,k,sizeof(this%cl_dxt),C_LOC(this%cl_dxt))
    if (err.ne.0) stop 'clSetKernelArg'
    err=clSetKernelArg(this%cl_cg_kernel,k+1,sizeof(this%cl_mask),C_LOC(this%cl_mask))
    if (err.ne.0) stop 'clSetKernelArg'
    err=clSetKernelArg(this%cl_cg_kernel,k+2,sizeof(this%cl_rtz1),C_LOC(this%cl_rtz1))
    if (err.ne.0) stop 'clSetKernelArg'
    err=clSetKernelArg(this%cl_cg_kernel,k+3,sizeof(this%cl_rtz2),C_LOC(this%cl_rtz2))
    if (err.ne.0) stop 'clSetKernelArg'
    err=clSetKernelArg(this%cl_cg_kernel,k+4,sizeof(this%cl_beta),C_LOC(this%cl_beta))
    if (err.ne.0) stop 'clSetKernelArg'
    err=clSetKernelArg(this%cl_cg_kernel,k+5,sizeof(this%n),C_LOC(this%n))
    if (err.ne.0) stop 'clSetKernelArg'
    err=clSetKernelArg(this%cl_cg_kernel,k+6,sizeof(this%cl_v),C_LOC(this%cl_b))
    if (err.ne.0) stop 'clSetKernelArg'
    err=clSetKernelArg(this%cl_cg_kernel,k+7,sizeof(this%cl_v),C_LOC(this%cl_gd))
    if (err.ne.0) stop 'clSetKernelArg'
    err=clSetKernelArg(this%cl_cg_kernel,k+8,sizeof(this%cl_v),C_LOC(this%cl_dg))
    if (err.ne.0) stop 'clSetKernelArg'
    err=clSetKernelArg(this%cl_cg_kernel,k+9,sizeof(this%cl_v),C_LOC(this%cl_v))
    if (err.ne.0) stop 'clSetKernelArg'

  end subroutine cg_banked_fpga_init_device


  subroutine cg_banked_fpga_populate(this, r, w, x, Xh,  c_Xh, bclst, gs_Xh, rtz1, beta)
    class(cg_banked_fpga_t), target :: this
    type(field_t), target :: r, w, x
    type(space_t), target :: Xh
    type(coef_t), target :: c_Xh
    type(bc_list_t) :: bclst
    type(gs_t), target :: gs_Xh
    real(kind=rp), target :: rtz1, beta
    integer :: err, k, lvl, id_tmp,  i
    integer, allocatable, target :: dg_tmp(:), gd_tmp(:)

    this%gs_m = gs_Xh%nlocal
    this%lo = gs_Xh%local_facet_offset
    this%nb = gs_Xh%nlocal_blks
    k = this%nbanks*11 +11
    err=clSetKernelArg(this%cl_cg_kernel,k,sizeof(this%gs_m),C_LOC(this%gs_m))
    if (err.ne.0) stop 'clSetKernelArg'
    err=clSetKernelArg(this%cl_cg_kernel,k+1,sizeof(this%lo),C_LOC(this%lo))
    if (err.ne.0) stop 'clSetKernelArg'
    err=clSetKernelArg(this%cl_cg_kernel,k+2,sizeof(this%nb),C_LOC(this%nb))
    if (err.ne.0) stop 'clSetKernelArg'

    call cl_write_banked_buffer(this%cmd_queue, this%array_size, c_Xh%dof%size(), c_Xh%mult, this%cl_mult)  
    call cl_write_banked_buffer(this%cmd_queue, this%array_size, c_Xh%dof%size(), x%x , this%cl_x)  
    call cl_write_banked_buffer(this%cmd_queue, this%array_size, c_Xh%dof%size(), w%x , this%cl_w)  
    call cl_write_banked_buffer(this%cmd_queue, this%array_size, c_Xh%dof%size(), r%x , this%cl_p)  
    call cl_write_banked_buffer(this%cmd_queue, this%array_size, c_Xh%dof%size(), r%x , this%cl_res)  
    call cl_write_banked_buffer(this%cmd_queue, this%array_size, c_Xh%dof%size(), c_Xh%g11, this%cl_g1)  
    call cl_write_banked_buffer(this%cmd_queue, this%array_size, c_Xh%dof%size(), c_Xh%g12, this%cl_g2)  
    call cl_write_banked_buffer(this%cmd_queue, this%array_size, c_Xh%dof%size(), c_Xh%g13, this%cl_g3)  
    call cl_write_banked_buffer(this%cmd_queue, this%array_size, c_Xh%dof%size(), c_Xh%g22, this%cl_g4)  
    call cl_write_banked_buffer(this%cmd_queue, this%array_size, c_Xh%dof%size(), c_Xh%g23, this%cl_g5)  
    call cl_write_banked_buffer(this%cmd_queue, this%array_size, c_Xh%dof%size(), c_Xh%g33, this%cl_g6)  
    
    err = clEnqueueWriteBuffer(this%cmd_queue,this%cl_dx,CL_TRUE,0_rp,this%dx_size,&
                               C_LOC(Xh%dx), 0,C_NULL_PTR,C_NULL_PTR)
    if (err.ne.0) stop 'clEnqueueWriteBuffer'
    err = clEnqueueWriteBuffer(this%cmd_queue,this%cl_dxt,CL_TRUE,0_rp,this%dx_size,&
                               C_LOC(Xh%dxt), 0,C_NULL_PTR,C_NULL_PTR)
    if (err.ne.0) stop 'clEnqueueWriteBuffer'
    err = clEnqueueWriteBuffer(this%cmd_queue,this%cl_mask,CL_TRUE,0_rp,int((bclst%bc(1)%bcp%msk(0)+1)*4,8),&
                               C_LOC(bclst%bc(1)%bcp%msk(0)), 0,C_NULL_PTR,C_NULL_PTR)
    if (err.ne.0) stop 'clEnqueueWriteBuffer'
    err = clEnqueueWriteBuffer(this%cmd_queue,this%cl_beta,CL_TRUE,0_rp,int(8,8),&
                               C_LOC(beta), 0,C_NULL_PTR,C_NULL_PTR)
    if (err.ne.0) stop 'clEnqueueWriteBuffer'
    err = clEnqueueWriteBuffer(this%cmd_queue,this%cl_rtz1,CL_TRUE,0_rp,int(8,8),&
                               C_LOC(rtz1), 0,C_NULL_PTR,C_NULL_PTR)
    if (err.ne.0) stop 'clEnqueueWriteBuffer'
    err = clEnqueueWriteBuffer(this%cmd_queue,this%cl_v,CL_TRUE,0_rp,int(this%gs_m*rp,8),&
                               C_LOC(gs_Xh%local_gs), 0,C_NULL_PTR,C_NULL_PTR)
    if (err.ne.0) stop 'clEnqueueWriteBuffer'
    allocate(dg_tmp(this%gs_m), gd_tmp(this%gs_m))
    do i = 1,this%gs_m
       lvl = mod(gs_Xh%local_gs_dof(i)-1, 32)
       id_tmp = (gs_Xh%local_gs_dof(i)-1)/32
       dg_tmp(i) = lvl/8
       gd_tmp(i) = mod(lvl, 8) + id_tmp*8
       print *, dg_tmp(i), gd_tmp(i)
    end do     
    err = clEnqueueWriteBuffer(this%cmd_queue,this%cl_dg,CL_TRUE,0_rp,int(this%gs_m*4,8),&
                               C_LOC(dg_tmp), 0,C_NULL_PTR,C_NULL_PTR)
    if (err.ne.0) stop 'clEnqueueWriteBuffer'
    err = clEnqueueWriteBuffer(this%cmd_queue,this%cl_gd,CL_TRUE,0_rp,int(this%gs_m*4,8),&
                               C_LOC(gd_tmp), 0,C_NULL_PTR,C_NULL_PTR)
    if (err.ne.0) stop 'clEnqueueWriteBuffer'
    err = clEnqueueWriteBuffer(this%cmd_queue,this%cl_b,CL_TRUE,0_rp,int(this%nb*4,8),&
                               C_LOC(gs_Xh%local_blk_len), 0,C_NULL_PTR,C_NULL_PTR)
    if (err.ne.0) stop 'clEnqueueWriteBuffer'
    deallocate(dg_tmp, gd_tmp)

    err=clFinish(this%cmd_queue)
    if (err.ne.0) stop 'clFinish'

  end subroutine cg_banked_fpga_populate 
  
  !> Deallocate a standard PCG solver
  subroutine cg_banked_fpga_free(this)
    class(cg_banked_fpga_t), intent(inout) :: this
    integer :: err

    call this%ksp_free()

    if (allocated(this%w)) then
       deallocate(this%w)
    end if

    if (allocated(this%r)) then
       deallocate(this%r)
    end if

    if (allocated(this%p)) then
       deallocate(this%p)
       err=clFinish(this%cmd_queue)
       if (err.ne.0) stop 'clFinish'
       err = clReleaseCommandQueue(this%cmd_queue) 
       err = clReleaseContext(this%context)  
       err = clReleaseMemObject(this%cl_x(1))
       if (err.ne.0) stop 'clRelasseMemObj'
       err = clReleaseMemObject(this%cl_w(1))
       if (err.ne.0) stop 'clRelasseMemObj'
       err = clReleaseMemObject(this%cl_g1(1))
       if (err.ne.0) stop 'clRelasseMemObj'
       err = clReleaseMemObject(this%cl_g2(1))
       if (err.ne.0) stop 'clRelasseMemObj'
       err = clReleaseMemObject(this%cl_g3(1))
       if (err.ne.0) stop 'clRelasseMemObj'
       err = clReleaseMemObject(this%cl_g4(1))
       if (err.ne.0) stop 'clRelasseMemObj'
       err = clReleaseMemObject(this%cl_g5(1))
       if (err.ne.0) stop 'clRelasseMemObj'
       err = clReleaseMemObject(this%cl_g6(1))
       if (err.ne.0) stop 'clRelasseMemObj'
    end if
   

  end subroutine cg_banked_fpga_free
  
  !> Standard Unpreconditioned CG solve
  function cg_banked_fpga_solve(this, Ax, x, f, n, coef, blst, gs_h, niter) result(ksp_results)
    class(cg_banked_fpga_t), intent(inout):: this
    class(ax_t), intent(inout) :: Ax
    type(field_t), intent(inout) :: x
    integer, intent(inout) :: n
    real(kind=rp), dimension(n), intent(inout) :: f
    type(coef_t), intent(inout) :: coef
    type(bc_list_t), intent(inout) :: blst
    type(gs_t), intent(inout) :: gs_h
    type(ksp_monitor_t) :: ksp_results
    integer, optional, intent(in) :: niter
    real(kind=rp), parameter :: one = 1.0
    real(kind=rp), parameter :: zero = 0.0
    integer :: iter, max_iter, err
    real(kind=rp), target :: rnorm, rtr, rtr0, rtz2, rtz1
    real(kind=rp), target :: beta, pap, alpha, alphm, eps, norm_fac
    
    if (present(niter)) then
       max_iter = niter
    else
       max_iter = KSP_MAX_ITER
    end if
    norm_fac = one/sqrt(coef%volume)

    rtz1 = one
    call rzero(x%x, n)
    call rzero(this%p, n)
    call copy(this%r, f, n)

    rtr = glsc3(this%r, coef%mult, this%r, n)
    rnorm = sqrt(rtr)*norm_fac
    ksp_results%res_start = rnorm
    ksp_results%res_final = rnorm
    ksp_results%iter = 0
    if(rnorm .eq. zero) return
    do iter = 1, max_iter
       err=clEnqueueTask(this%cmd_queue,this%cl_cg_kernel,0,C_NULL_PTR,C_NULL_PTR)
       if (err .ne. 0) stop 'clEnqueueEnqueueTask cg'
       call host_kernel(this, gs_h, blst, n)
    end do
    err=clFinish(this%cmd_queue)
    ksp_results%res_final = rnorm
    ksp_results%iter = iter
  end function cg_banked_fpga_solve
 
  subroutine host_kernel(this, gs_h, blst, n)
    class(cg_banked_fpga_t), target :: this
    type(gs_t) :: gs_h
    type(bc_list_t) :: blst
    integer :: n, err
    real(kind=rp), target :: rtz1, rnorm
    err = clEnqueueReadBuffer(this%cmd_queue,this%cl_rtz1,CL_TRUE,&
                                 0_rp,int(rp,8),C_LOC(rtz1),0,C_NULL_PTR,C_NULL_PTR)
    !if (err.ne.0) stop 'clEnqueueReadBuffer'
    !err = clEnqueueReadBuffer(this%cmd_queue,this%cl_w,CL_TRUE,&
    !                             0_rp,this%array_size,C_LOC(this%w),0,C_NULL_PTR,C_NULL_PTR)
    !if (err.ne.0) stop 'clEnqueueReadBuffer'
    err=clFinish(this%cmd_queue)
    rnorm = sqrt(rtz1)
    print *, rnorm, rtz1
    !call gs_op_vector(gs_h, this%w, n, GS_OP_ADD)
    !err=clEnqueueWriteBuffer(this%cmd_queue,this%cl_w,CL_TRUE,0_rp,&
    !                         this%array_size,C_LOC(this%w), 0,C_NULL_PTR,C_NULL_PTR)
    if (err.ne.0) stop 'clEnqueueWriteBuffer cg'    
  end subroutine host_kernel
  subroutine banked_fpga_get_data(this, x)
    type(cg_banked_fpga_t), target :: this
    type(field_t), target :: x
    integer :: err, n, i, b, j
    real(kind=rp), allocatable, target :: temp(:)
    n = x%dof%size()
    allocate(temp(n))
    err = clEnqueueReadBuffer(this%cmd_queue,this%cl_x(1),CL_TRUE,&
                                 0_rp,this%array_size/4,C_LOC(temp),0,C_NULL_PTR,C_NULL_PTR)
    err = clEnqueueReadBuffer(this%cmd_queue,this%cl_x(2),CL_TRUE,&
                                 0_rp,this%array_size/4,C_LOC(temp(n/4+1)),0,C_NULL_PTR,C_NULL_PTR)
    err = clEnqueueReadBuffer(this%cmd_queue,this%cl_x(3),CL_TRUE,&
                                 0_rp,this%array_size/4,C_LOC(temp(n/2+1)),0,C_NULL_PTR,C_NULL_PTR)
    err = clEnqueueReadBuffer(this%cmd_queue,this%cl_x(4),CL_TRUE,&
                                 0_rp,this%array_size/4,C_LOC(temp(3*n/4+1)),0,C_NULL_PTR,C_NULL_PTR)
    if (err.ne.0) stop 'clEnqueueReadBuffer'
    do i = 1, n/(4*8)
       do b = 1, this%nbanks
          do j = 1, 8
             x%x((i-1)*32+(b-1)*8+j,1,1,1) = temp((b-1)*n/4+(i-1)*8+j)
          end do
       end do
    end do
    deallocate(temp)
 
   end subroutine banked_fpga_get_data 

   subroutine cl_create_banked_buffer(context, array_size, CL_MEM_RW, cl_x)
     integer(c_intptr_t), intent(inout) :: context
     integer(c_intptr_t), intent(inout) :: cl_x(4)
     integer(c_size_t) :: array_size, bank_array_size
     integer(c_int64_t) :: CL_MEM_RW
     integer :: err
     bank_array_size = array_size/4

     cl_x(1) = clCreateBuffer(context,ior(CL_MEM_RW,CL_CHANNEL_1_INTELFPGA),&
                          bank_array_size,C_NULL_PTR, err)
    if (err.ne.0) stop 'clCreateBuffer'
     cl_x(2) = clCreateBuffer(context,ior(CL_MEM_RW,CL_CHANNEL_2_INTELFPGA),&
                          bank_array_size,C_NULL_PTR, err)
    if (err.ne.0) stop 'clCreateBuffer'
     cl_x(3) = clCreateBuffer(context,ior(CL_MEM_RW,CL_CHANNEL_3_INTELFPGA),&
                          bank_array_size,C_NULL_PTR, err)
    if (err.ne.0) stop 'clCreateBuffer'
     cl_x(4) = clCreateBuffer(context,ior(CL_MEM_RW,CL_CHANNEL_4_INTELFPGA),&
                          bank_array_size,C_NULL_PTR, err)
    if (err.ne.0) stop 'clCreateBuffer'
  end subroutine cl_create_banked_buffer 
  subroutine cl_write_banked_buffer(cmd_queue, array_size, n, x, cl_x)
    integer(c_intptr_t), intent(inout) :: cmd_queue
    integer(c_intptr_t), intent(inout) :: cl_x(4)
    integer(c_size_t) :: array_size, bank_array_size
    integer :: n
    real(kind=rp), target :: x(n)
    real(kind=rp), allocatable, target :: temp(:)
    integer :: err, n_bank_array, i, nbanks, b, j
    nbanks = 4
    bank_array_size = array_size/nbanks
    n_bank_array = n/nbanks
    allocate(temp(n_bank_array))
    do b = 1, nbanks
       do i = 1, n_bank_array/8
          do j = 1, 8
             temp((i-1)*8+j) = x((i-1)*nbanks*8+j+(b-1)*8)
          end do
       end do
       err = clEnqueueWriteBuffer(cmd_queue,cl_x(b),CL_TRUE,0_rp,bank_array_size,&
                               C_LOC(temp), 0,C_NULL_PTR,C_NULL_PTR)
       if (err.ne.0) stop 'clEnqueueWriteBuffer'
    end do
    deallocate(temp)
  end subroutine cl_write_banked_buffer 
  subroutine cl_set_kernel_arg(kernel, cl_ptr, arg_id)
    integer(c_intptr_t), target, intent(inout) :: cl_ptr
    integer(c_intptr_t) :: kernel
    integer :: arg_id, err

    err=clSetKernelArg(kernel,arg_id,sizeof(cl_ptr),C_LOC(cl_ptr))
    if (err.ne.0) stop 'clSetKernelArg'
  end subroutine cl_set_kernel_arg
       
end module cg_banked_fpga
  

