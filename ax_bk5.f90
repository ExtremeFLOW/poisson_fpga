module ax_bk5
  use ax_product
  use clfortran
  implicit none

  type, public, extends(ax_t) :: ax_bk5_t
   contains
     procedure, nopass :: compute => ax_bk5_compute
  end type ax_bk5_t

  integer(c_intptr_t), target :: cmd_queue, cl_bk5_kernel

contains

  subroutine ax_bk5_compute(w, u, coef, msh, Xh)
    type(mesh_t), intent(inout) :: msh
    type(space_t), intent(inout) :: Xh
    type(coef_t), intent(inout) :: coef
    real(kind=rp), intent(inout) :: w(Xh%lx, Xh%ly, Xh%lz, msh%nelv)
    real(kind=rp), intent(inout) :: u(Xh%lx, Xh%ly, Xh%lz, msh%nelv)
   
    integer(c_int32_t) :: err

    err=clEnqueueTask(cmd_queue,cl_bk5_kernel,0,C_NULL_PTR,C_NULL_PTR)
    if (err .ne. 0) stop 'clEnqueueEnqueueTask'
  end subroutine ax_bk5_compute
end module ax_bk5
