package com.qczy.controller.dept;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.common.constant.SystemConstant;
import com.qczy.common.log.BusinessType;
import com.qczy.common.log.Log;
import com.qczy.common.result.Result;
import com.qczy.model.entity.DeptEntity;
import com.qczy.model.request.DeleteRequest;
import com.qczy.service.DeptService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.util.ObjectUtils;
import org.springframework.validation.BindingResult;
import org.springframework.web.bind.annotation.*;

import javax.validation.Valid;
import java.util.Objects;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/6 14:56
 * @Description:
 */
@RestController
@RequestMapping("/dept")
@Api(tags = "部门管理")
public class DeptController {

    @Autowired
    private DeptService deptService;

    @GetMapping("/getDeptSelect")
    @ApiOperation("获取所有部门（不包含分页信息）")
    public Result getDeptSelect() {
        return Result.ok(deptService.getDeptSelect());
    }


    /**
     * 查询【请填写功能名称】列表
     */
    @GetMapping("/list")
    @ApiOperation("获取带分页带条件查询")
    public Result list(@RequestParam Integer page,
                       @RequestParam Integer limit,
                       @ModelAttribute DeptEntity request) {
        Page<DeptEntity> pageParam = new Page<>(page, limit);
        IPage<DeptEntity> deptEntityList = deptService.selectDeptList(pageParam, request);
        return Result.ok(deptEntityList);
    }


    /**
     * 新增保存【请填写功能名称】
     */

    @PostMapping("/add")
    @ApiOperation("新增")
    @Log(title = "部门管理", businessType = BusinessType.INSERT)
    public Result addSave(@RequestBody @Valid DeptEntity dept, BindingResult bindingResult) {
        if (bindingResult.hasErrors()) {
            return Result.fail(Objects.requireNonNull(bindingResult.getFieldError()).getDefaultMessage());
        }
        return Result.ok(deptService.insertDept(dept));
    }


    /**
     * 修改保存【请填写功能名称】
     */

    @PostMapping("/edit")
    @ApiOperation("修改")
    @Log(title = "部门管理", businessType = BusinessType.UPDATE)
    public Result editSave(@RequestBody @Valid DeptEntity dept, BindingResult bindingResult) {
        if (bindingResult.hasErrors()) {
            return Result.fail(Objects.requireNonNull(bindingResult.getFieldError()).getDefaultMessage());
        }
        return Result.ok(deptService.updateDept(dept));
    }

    /**
     * 删除【请填写功能名称】
     */

    @DeleteMapping("/remove")
    @ApiOperation("批量删除")
    @Log(title = "部门管理", businessType = BusinessType.DELETE)
    public Result remove(@RequestBody DeleteRequest request) {
        if (ObjectUtils.isEmpty(request) && request.getIds() == null) {
            return Result.ok("请传入要删除的数据！");
        }
        // 判断有没有用户绑定
        if (deptService.getDeptAndUserCount(request.getIds()) >= SystemConstant.MAX_SIZE) {
            return Result.fail("当前角色已被用户绑定，请先修改用户后重试！");
        }

        return Result.ok(deptService.deleteDeptByIds(request.getIds()));
    }


    /**
     *  获取所有部门下的用户
     */
    @GetMapping("/getDeptByUserList")
    @ApiOperation("获取所有部门下的用户")
    public Result getDeptByUserList(){
        return Result.ok(deptService.getDeptByUserList());
    }

}
