package com.qczy.controller.menu;

import com.qczy.common.result.Result;
import com.qczy.service.MenuService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/7/26 11:46
 * @Description:
 */
@RestController
@RequestMapping("/menu")
@Api(tags = "菜单管理")
public class MenuController {


    @Autowired
    private MenuService menuService;


    // 构建树形菜单
    @GetMapping("/getMenuTree")
    @ApiOperation("构建树形菜单列表")
    public Result getTreeMenuList() {
        return Result.ok(menuService.getTreeMenuList());
    }


    // 构建按钮树形菜单
    @GetMapping("/getMenuButtonTree")
    @ApiOperation("构建按钮树形菜单")
    public Result getButtonTreeMenuList() {
        return Result.ok(menuService.getMenuButtonTree());
    }







    /* *//**
     * 新增保存
     *//*
    @PostMapping("/add")
    @ResponseBody
    public Result addSave(MenuEntity menu) {
        return Result.ok(menuService.insertMenu(menu));
    }


    *//**
     * 修改保存
     *//*
    @PostMapping("/edit")
    @ResponseBody
    public Result editSave(MenuEntity menu) {
        return Result.ok(menuService.updateMenu(menu));
    }

    *//**
     * 删除
     *//*
    @PostMapping("/remove")
    @ResponseBody
    public Result remove(String ids) {
        return Result.ok(menuService.deleteMenuByIds(ids));
    }*/
}
