package com.qczy.service;

import java.util.List;
import java.util.Map;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.qczy.model.entity.UserEntity;
import com.qczy.model.request.DeleteRequest;
import com.qczy.model.request.PasswordRequest;
import com.qczy.model.request.UserLoginRequest;
import com.qczy.model.request.UserRequest;

import javax.servlet.http.HttpServletRequest;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/7/15 15:47
 * @Description:
 */
public interface UserService extends IService<UserEntity> {

    List<UserEntity> getUserSelect();

    Map<String, Object> login(UserLoginRequest loginRequest, HttpServletRequest request);


    /**
     * 查询【请填写功能名称】
     *
     * @param id 【请填写功能名称】主键
     * @return 【请填写功能名称】
     */
    public UserEntity selectUserById(Integer id);

    /**
     * 查询【请填写功能名称】列表
     *
     * @param pageParam 分页信息，【请填写功能名称】
     * @param request   查询参数，【请填写功能名称】
     * @return 【请填写功能名称】集合
     */
    IPage<UserEntity> selectUserList(Page<UserEntity> pageParam, UserRequest request);

    /**
     * 获取单个用户【请填写功能名称】信息
     *
     * @param id 【请填写功能名称】主键
     * @return 结果
     */
    UserEntity getUserById(Integer id);


    /**
     * 新增【请填写功能名称】
     *
     * @param user 【请填写功能名称】
     * @return 结果
     */
    public int insertUser(UserEntity user);

    /**
     * 修改【请填写功能名称】
     *
     * @param user 【请填写功能名称】
     * @return 结果
     */
    public int updateUser(UserEntity user);

    /**
     * 批量删除【请填写功能名称】
     *
     * @param ids 需要删除的【请填写功能名称】主键集合
     * @return 结果
     */
    public int deleteUserByIds(String ids);

    /**
     * 删除【请填写功能名称】信息
     *
     * @param id 【请填写功能名称】主键
     * @return 结果
     */
    public int deleteUserById(Integer id);


    /**
     * 我的删除【请填写功能名称】信息
     *
     * @param ids 【请填写功能名称】主键
     * @return 结果
     */
    int MyDeleteAll(int[] ids);


    /**
     * 根据用户名查询数量
     */
    int getByUsernameCount(String username);
    /**
     * 根据id,用户名查询 数量
     */
    int getByUsernameCount(Integer id, String username);

    String setPassword(String pwd);

    /**
     * 重置id
     */
    int resetPassword(PasswordRequest request);

    /**
     *  判断当前用户 是否 在多人任务-团队中，如果在，则返回 true
     */
    boolean isManyTeamUser(DeleteRequest deleteRequest);

    /**
     * 重置默认密码
     */
    int resetDefaultPassword(Integer id);
}
