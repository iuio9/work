package com.qczy.controller.role;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.common.constant.SystemConstant;
import com.qczy.common.log.BusinessType;
import com.qczy.common.log.Log;
import com.qczy.common.result.Result;
import com.qczy.model.entity.RoleEntity;
import com.qczy.model.request.DeleteRequest;
import com.qczy.model.request.RoleMenuRequest;
import com.qczy.service.RoleService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import org.apache.commons.io.FileUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.util.ObjectUtils;
import org.springframework.util.StringUtils;
import org.springframework.web.bind.annotation.*;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/1 10:01
 * @Description:
 */
@RestController
@RequestMapping("/role")
@Api(tags = "角色管理")
public class RoleController {

    @Autowired
    private RoleService roleService;


    @GetMapping("/getRoleSelect")
    @ApiOperation("获取所有角色（不包含分页信息）")
    public Result getRoleSelect() {
        return Result.ok(roleService.getRoleSelect());
    }


    /**
     * 查询【请填写功能名称】列表
     */
    @GetMapping("/list")
    @ApiOperation("获取带分页带条件查询")
    public Result list(@RequestParam Integer page,
                       @RequestParam Integer limit,
                       @ModelAttribute RoleEntity role) {
        Page<RoleEntity> pageParam = new Page<>(page, limit);
        IPage<RoleEntity> roleEntityList = roleService.selectRoleList(pageParam, role);
        return Result.ok(roleEntityList);
    }


    /**
     * 新增保存【请填写功能名称】
     */
    @PostMapping("/add")
    @ApiOperation("新增")
    @Log(title = "角色管理", businessType = BusinessType.INSERT)
    public Result addSave(@RequestBody RoleEntity role) {
        if (roleService.getByRoleNameCount(role.getRoleName()) >= SystemConstant.MAX_SIZE) {
            return Result.fail("角色名已存在");
        }
        return Result.ok(roleService.insertRole(role));
    }


    /**
     * 修改保存【请填写功能名称】
     */
    @PostMapping("/edit")
    @ApiOperation("修改")
    @Log(title = "角色管理", businessType = BusinessType.UPDATE)
    public Result editSave(@RequestBody RoleEntity role) {
        if (roleService.getByRoleNameCount(role.getId(), role.getRoleName()) >= SystemConstant.MAX_SIZE) {
            return Result.fail("用户名已存在");
        }
        return Result.ok(roleService.updateRole(role));
    }

    /**
     * 删除【请填写功能名称】
     */
    @DeleteMapping("/remove")
    @ApiOperation("批量删除")
    @Log(title = "角色管理", businessType = BusinessType.DELETE)
    public Result remove(@RequestBody DeleteRequest request) {
        if (ObjectUtils.isEmpty(request) && request.getIds() == null) {
            return Result.ok("请传入要删除的数据！");
        }
        // 判断有没有用户绑定
        if (roleService.getRoleAndUserCount(request.getIds()) >= SystemConstant.MAX_SIZE) {
            return Result.fail("当前角色已被用户绑定，请先修改用户后重试！");
        }

        return Result.ok(roleService.MyDeleteAll(request.getIds()));
    }

    /**
     * 新增角色所需要的权限
     */

    @PostMapping("/addRoleMenu")
    @ApiOperation("新增角色所需要的权限")
    public Result addRoleMenu(@RequestBody RoleMenuRequest request) {
        if (ObjectUtils.isEmpty(request)) {
            return Result.fail("对象不能为空！");
        }

        if (StringUtils.isEmpty(request.getRoleId())) {
            return Result.fail("角色id不能为空！");
        }

        if (request.getMenuIds() == null || request.getMenuIds().isEmpty()) {
            return Result.fail("菜单id不能为空！");
        }
        return Result.ok(roleService.addRoleMenu(request));
    }


    /**
     *  新增角色所需要的按钮权限
     */
    @PostMapping("/addRoleButton")
    @ApiOperation("新增角色所需要的权限")
    public Result addRoleButton(@RequestBody RoleMenuRequest request) {
        if (ObjectUtils.isEmpty(request)) {
            return Result.fail("对象不能为空！");
        }

        if (StringUtils.isEmpty(request.getRoleId())) {
            return Result.fail("角色id不能为空！");
        }

        if (request.getMenuIds() == null || request.getMenuIds().isEmpty()) {
            return Result.fail("按钮id不能为空！");
        }
        return Result.ok(roleService.addRoleButton(request));
    }

    /**
     * 回显角色所需要的权限
     */
    @GetMapping("/getRoleMenu")
    @ApiOperation("获取当前角色所需要的权限")
    public Result getRoleMenu(@RequestParam Integer roleId) {
        if (StringUtils.isEmpty(roleId)) {
            return Result.fail("角色id不能为空！");
        }
        return Result.ok(roleService.getRoleMenu(roleId));
    }


    /**
     * 回显角色所需要的权限
     */
    @GetMapping("/getRoleButton")
    @ApiOperation("获取当前角色所需要的权限")
    public Result getRoleButton(@RequestParam Integer roleId) {
        if (StringUtils.isEmpty(roleId)) {
            return Result.fail("角色id不能为空！");
        }
        return Result.ok(roleService.getRoleButton(roleId));
    }

    /**
     *  判断这个当前用户是否是多人标注权限
     */
    @GetMapping("/isManyTask")
    public Result isManyTask() {
        return Result.ok(roleService.isManyTask());
    }


}
