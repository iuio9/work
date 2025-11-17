package com.qczy.controller.user;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.common.constant.SystemConstant;
import com.qczy.common.log.BusinessType;
import com.qczy.common.log.Log;
import com.qczy.common.result.Result;
import com.qczy.common.result.ResultCodeEnum;
import com.qczy.model.entity.RoleEntity;
import com.qczy.model.entity.UserEntity;
import com.qczy.model.request.DeleteRequest;
import com.qczy.model.request.PasswordRequest;
import com.qczy.model.request.UserLoginRequest;
import com.qczy.model.request.UserRequest;
import com.qczy.model.response.RoleResponse;
import com.qczy.model.response.RouterResponse;
import com.qczy.service.MenuService;
import com.qczy.service.RoleService;
import com.qczy.service.UserService;
import com.qczy.utils.JwtTokenUtil;

import java.lang.String;

import io.swagger.annotations.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.util.ObjectUtils;
import org.springframework.util.StringUtils;
import org.springframework.validation.BindingResult;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

import javax.servlet.http.HttpServletRequest;
import javax.validation.Valid;
import javax.validation.constraints.NotEmpty;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/7/15 11:49
 * @Description: 用户管理
 */
@RestController
@Api(tags = "用户管理")
public class UserController {


    private static final Logger logger = LoggerFactory.getLogger(UserController.class);

    @Autowired
    private UserService userService;
    @Autowired
    private RoleService roleService;
    @Autowired
    private MenuService menuService;
    @Autowired
    private PasswordEncoder passwordEncoder;
    @Autowired
    private JwtTokenUtil jwtTokenUtil;


    @GetMapping("/user/getUserSelect")
    @ApiOperation("获取所有用户，（不包含冻结的用户）")
    public Result getUserSelect() {
        return Result.ok(userService.getUserSelect());
    }


    @PostMapping("/auth/login")
    @ApiOperation("登录")
    public Result login(@RequestBody @NotEmpty UserLoginRequest loginRequest, HttpServletRequest request) {
        UserEntity user = userService.getOne(
                new LambdaQueryWrapper<UserEntity>()
                        .eq(UserEntity::getUserName, loginRequest.getUserName())
                        .eq(UserEntity::getIsDeleted, SystemConstant.SYSTEM_NO_DISABLE)
        );
        if (ObjectUtils.isEmpty(user))
            return Result.fail(ResultCodeEnum.LOGIN_MOBLE_ERROR.getMessage());
        if (StringUtils.isEmpty(user.getStatus()) || user.getStatus().equals(SystemConstant.SYSTEM_YES_FREEZE))
            return Result.fail(ResultCodeEnum.ACCOUNT_STOP.getMessage());
        if (!passwordEncoder.matches(loginRequest.getPassword(), user.getPassword()))
            return Result.fail(ResultCodeEnum.PASSWORD_ERROR.getMessage());

        Map<String, Object> data = userService.login(loginRequest, request);
        return Result.ok(data);
    }


    @GetMapping("/auth/getUserInfo")
    @ApiOperation("获取用户信息")
    @ApiImplicitParam(name = "Authorization", value = "Authorization", required = true, dataType = "String", paramType = "header")
    @SuppressWarnings({"all"})
    public Result getUserInfo(HttpServletRequest request) {
        String token = request.getHeader("Authorization");
        if (StringUtils.isEmpty(token))
            return Result.fail(ResultCodeEnum.TOKEN_NOT_LEGAL.getMessage());
        logger.info("token->" + token);
        //解析token,获取用户账号
        String[] s1 = token.split(" ");

        String username = jwtTokenUtil.getUserNameFromToken(s1[1]);
        if (StringUtils.isEmpty(username))
            return Result.fail(ResultCodeEnum.TOKEN_NOT_LEGAL.getMessage());
        UserEntity user = userService.getOne(
                new LambdaQueryWrapper<UserEntity>()
                        .eq(UserEntity::getUserName, username)
                        .eq(UserEntity::getIsDeleted, SystemConstant.SYSTEM_NO_DISABLE)
        );
        for (String roleIdStr : user.getUserRoles().split(",")) {
            if (roleIdStr.equals("1")){
                user.setIsHide(1);
                break;
            }else {
                user.setIsHide(0);
            }
        }


/*        //根据账号所拥有的角色和菜单权限
        Map<String, Object> map = new HashMap<>();
        //用户昵称
        map.put("nickname", user.getNickName());
        //角色信息
        RoleEntity roleEntity = roleService.getById(user.getUserRoles());
        RoleResponse role = new RoleResponse();
        BeanUtils.copyProperties(roleEntity, role);
        map.put("role", role);*/
        //菜单信息（树化）
        // List<RouterResponse> routerVoList = menuService.getUserMenuList(roleEntity.getId());
        //map.put("routes",routerVoList);
        return Result.ok(user);
    }

    @GetMapping("/route/getUserRoutes")
    @ApiOperation("获取用户信息")
    @ApiImplicitParam(name = "Authorization", value = "Authorization", required = true, dataType = "String", paramType = "header")
    @SuppressWarnings({"all"})
    public Result getUserRoutes(HttpServletRequest request) {
        String token = request.getHeader("Authorization");
        String[] s1 = token.split(" ");
        if (StringUtils.isEmpty(token))
            return Result.fail(ResultCodeEnum.TOKEN_NOT_LEGAL.getMessage());
        logger.info("token->" + token);
        //解析token,获取用户账号
        String username = jwtTokenUtil.getUserNameFromToken(s1[1]);
        if (StringUtils.isEmpty(username))
            return Result.fail(ResultCodeEnum.TOKEN_NOT_LEGAL.getMessage());
        UserEntity user = userService.getOne(
                new LambdaQueryWrapper<UserEntity>()
                        .eq(UserEntity::getUserName, username)
                        .eq(UserEntity::getIsDeleted, SystemConstant.SYSTEM_NO_DISABLE)
        );
        logger.info("username->" + username);

        //根据账号所拥有的角色和菜单权限
        Map<String, Object> map = new HashMap<>();
        //用户昵称
        map.put("nickname", user.getNickName());
        //角色信息
        //RoleEntity roleEntity = roleService.getById(user.getUserRoles());
        if (StringUtils.isEmpty(user.getUserRoles())) {
            String[] roleid = user.getUserRoles().split(",");
            RoleEntity roleEntity = roleService.getById(roleid);
            RoleResponse role = new RoleResponse();
            BeanUtils.copyProperties(roleEntity, role);
        }

        //菜单信息（树化）
        List<RouterResponse> routerVoList = menuService.getUserMenuList(user.getUserRoles());
        map.put("routes", routerVoList);
        return Result.ok(map);
    }

    /**
     * 查询【请填写功能名称】列表
     */

    //获取带分页带条件查询
    @GetMapping("/user/list")
    @ApiOperation("获取带分页带条件查询")
    public Result list(
            @RequestParam Integer page,
            @RequestParam Integer limit,
            @ModelAttribute UserRequest request) {
        Page<UserEntity> pageParam = new Page<>(page, limit);
        IPage<UserEntity> userEntityList = userService.selectUserList(pageParam, request);
        return Result.ok(userEntityList);

    }


    /**
     * 查询单个用户【请填写功能名称】
     */
    @GetMapping("/user/getUserById")
    @ApiOperation("查询单个用户")
    public Result getUserById(@RequestParam Integer id) {
        return Result.ok(userService.getUserById(id));
    }

    /**
     * 新增保存【请填写功能名称】
     */
    @PostMapping("/user/add")
    @ApiOperation("新增")
    @Log(title = "用户管理", businessType = BusinessType.INSERT)
    public Result addSave(@RequestBody @Validated UserEntity user) {
        if (userService.getByUsernameCount(user.getUserName()) >= SystemConstant.MAX_SIZE) {
            return Result.fail("用户名已存在");
        }
        return Result.ok(userService.insertUser(user));
    }

    /**
     * 修改保存【请填写功能名称】
     */
    @PostMapping("/user/edit")
    @ApiOperation("修改")
    @Log(title = "用户管理", businessType = BusinessType.UPDATE)
    public Result editSave(@RequestBody @Validated UserEntity user) {
        if (userService.getByUsernameCount(user.getId(), user.getUserName()) >= SystemConstant.MAX_SIZE) {
            return Result.fail("用户名已存在");
        }
        return Result.ok(userService.updateUser(user));
    }


    /**
     * 删除【请填写功能名称】
     */
    @DeleteMapping("/user/remove")
    @ApiOperation("批量删除")
    @Log(title = "用户管理", businessType = BusinessType.DELETE)
    public Result remove(@RequestBody DeleteRequest request) {
        if (ObjectUtils.isEmpty(request) && request.getIds() == null) {
            return Result.fail("请选择要删除的数据！");
        }
        if (userService.isManyTeamUser(request)) {
            return Result.fail("当前用户正在多人团队中，请先在多人团队中删除后，重试！");
        }
        return Result.ok(userService.MyDeleteAll(request.getIds()));
    }

    /**
     * 重置密码
     */
    @PostMapping("/user/resetPassword")
    public Result resetPassword(@RequestBody @Valid PasswordRequest request, BindingResult bindingResult) {
        if (bindingResult.hasErrors()) {
            return Result.fail(Objects.requireNonNull(bindingResult.getFieldError()).getDefaultMessage());
        }
        if (!request.getPassword().equals(request.getConfirmPassword())) {
            return Result.fail("二次密码不正确！");
        }
        int result = userService.resetPassword(request);
        if (result == 1) {
            return Result.ok(1);
        } else {
            return Result.fail("重置密码失败！");
        }

    }


    /**
     *
     * 重置默认密码
     */
    @GetMapping("/user/resetDefaultPassword")
    public Result resetDefaultPassword(Integer id) {
        if (id == null){
            return Result.fail("用户id不能为空！");
        }
        int result = userService.resetDefaultPassword(id);
        if (result > 0 ) {
            return Result.ok(1);
        }else {
            return Result.fail("用户密码重置失败！");
        }
    }


    @GetMapping("/setPassword")
    public Result setPassword(String pwd) {
        return Result.ok(userService.setPassword(pwd));
    }

}
