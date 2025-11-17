package com.qczy.service.impl;

import cn.hutool.core.convert.Convert;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.qczy.common.constant.SystemConstant;
import com.qczy.mapper.LoginLogMapper;
import com.qczy.mapper.TeamUserMapper;
import com.qczy.model.entity.LoginLogEntity;
import com.qczy.model.entity.TeamUserEntity;
import com.qczy.model.entity.UserEntity;
import com.qczy.mapper.UserMapper;
import com.qczy.model.request.DeleteRequest;
import com.qczy.model.request.PasswordRequest;
import com.qczy.model.request.UserLoginRequest;
import com.qczy.model.request.UserRequest;
import com.qczy.service.UserService;
import com.qczy.utils.IpUtils;
import com.qczy.utils.JwtTokenUtil;
import org.springframework.beans.factory.annotation.Autowired;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.util.ObjectUtils;
import org.springframework.util.StringUtils;

import javax.servlet.http.HttpServletRequest;
import java.util.*;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/7/15 15:47
 * @Description:
 */
@Service
public class UserServiceImpl extends ServiceImpl<UserMapper, UserEntity> implements UserService {

    @Autowired
    private UserMapper userMapper;


    @Autowired
    private UserDetailsService userDetailsService;

    @Autowired
    private JwtTokenUtil jwtTokenUtil;

    @Autowired
    private PasswordEncoder passwordEncoder;

    @Autowired
    private LoginLogMapper loginLogMapper;

    @Autowired
    private TeamUserMapper teamUserMapper;


    @Value("${jwt.tokenHead}")
    private String tokenHead;


    @Override
    public List<UserEntity> getUserSelect() {
        return userMapper.selectList(new LambdaQueryWrapper<UserEntity>()
                .eq(UserEntity::getStatus, SystemConstant.SYSTEM_NO_FREEZE)
                .eq(UserEntity::getIsDeleted, SystemConstant.SYSTEM_NO_FREEZE));
    }

    @Override
    public Map<String, Object> login(UserLoginRequest loginRequest, HttpServletRequest request) {
        String token = null;

        UserDetails userDetails = userDetailsService.loadUserByUsername(loginRequest.getUserName());
        UsernamePasswordAuthenticationToken authentication = new UsernamePasswordAuthenticationToken(userDetails, null, null);
        SecurityContextHolder.getContext().setAuthentication(authentication);
        token = jwtTokenUtil.generateToken(userDetails);
        Map<String, Object> map = new HashMap<>();
        map.put("token", token);
        map.put("tokenHead", tokenHead);
        map.put("name", userDetails.getUsername());


        // 记录登陆日志
        setLoginLog(loginRequest.getUserName(), request);

        return map;

    }

    private void setLoginLog(String userName, HttpServletRequest request) {
        UserEntity user = userMapper.selectOne(
                new LambdaQueryWrapper<UserEntity>()
                        .eq(UserEntity::getUserName, userName)
                        .eq(UserEntity::getIsDeleted, SystemConstant.SYSTEM_NO_DISABLE)
        );

        LoginLogEntity loginLog = new LoginLogEntity();
        loginLog.setLoginName(user.getUserName());
        loginLog.setIpaddr(IpUtils.getLocalIP());
        // 获取客户端操作系统
        loginLog.setOs(IpUtils.getOsName(request));
        // 获取客户端浏览器
        loginLog.setBrowser(IpUtils.getBrowserName(request));
        loginLog.setLoginTime(new Date());
        loginLogMapper.insert(loginLog);
    }


    /**
     * 查询【请填写功能名称】
     *
     * @param id 【请填写功能名称】主键
     * @return 【请填写功能名称】
     */
    @Override
    public UserEntity selectUserById(Integer id) {
        return userMapper.selectUserById(id);
    }

    /**
     * 查询【请填写功能名称】列表
     *
     * @param pageParam 分页信息，【请填写功能名称】
     * @param request   查询参数，【请填写功能名称】
     * @return 【请填写功能名称】
     */
    @Override
    public IPage<UserEntity> selectUserList(Page<UserEntity> pageParam, UserRequest request) {
        return userMapper.selectUserList(pageParam, request);
    }

    @Override
    public UserEntity getUserById(Integer id) {
        return userMapper.selectById(id);
    }


    /**
     * 新增【请填写功能名称】
     *
     * @param user 【请填写功能名称】
     * @return 结果
     */
    @Override
    public int insertUser(UserEntity user) {
        user.setCreateTime(new Date());
        user.setIsDeleted(SystemConstant.SYSTEM_NO_DISABLE);
        // 密码加密
        user.setPassword(passwordEncoder.encode(user.getPassword()));
        user.setIsAllowDeletion(SystemConstant.YES_DISABLE_DATA);
        return userMapper.insertUser(user);
    }

    /**
     * 修改【请填写功能名称】
     *
     * @param user 【请填写功能名称】
     * @return 结果
     */
    @Override
    public int updateUser(UserEntity user) {
        user.setUpdateTime(new Date());
        return userMapper.updateUser(user);
    }

    /**
     * 批量删除【请填写功能名称】
     *
     * @param ids 需要删除的【请填写功能名称】主键
     * @return 结果
     */
    @Override
    public int deleteUserByIds(String ids) {
        //return userMapper.deleteUserByIds(Convert.toStrArray(ids));
        if (!StringUtils.isEmpty(ids)) {
            for (Integer id : Convert.toIntArray(ids)) {
                // 更改状态
                UserEntity user = userMapper.selectById(id);
                user.setIsDeleted(1);
                userMapper.updateById(user);
            }
            return 1;
        }
        return 0;
    }

    /**
     * 删除【请填写功能名称】信息
     *
     * @param id 【请填写功能名称】主键
     * @return 结果
     */
    @Override
    public int deleteUserById(Integer id) {
        return userMapper.deleteUserById(id);
    }

    @Override
    public int MyDeleteAll(int[] ids) {
        for (int id : ids) {
            UserEntity user = userMapper.selectById(id);
            user.setIsDeleted(SystemConstant.SYSTEM_YES_DISABLE);
            userMapper.updateById(user);
        }
        return 1;
    }

    @Override
    public int getByUsernameCount(String username) {
        return userMapper.selectCount(
                new LambdaQueryWrapper<UserEntity>()
                        .eq(UserEntity::getUserName, username)
                        .eq(UserEntity::getIsDeleted, SystemConstant.SYSTEM_NO_DISABLE)
        );
    }

    @Override
    public int getByUsernameCount(Integer id, String username) {
        return userMapper.selectCount(
                new LambdaQueryWrapper<UserEntity>()
                        .ne(UserEntity::getId, id)
                        .eq(UserEntity::getUserName, username)
                        .eq(UserEntity::getIsDeleted, SystemConstant.SYSTEM_NO_DISABLE)
        );
    }

    @Override
    public String setPassword(String pwd) {
        return passwordEncoder.encode(pwd);
    }

    @Override
    public int resetPassword(PasswordRequest request) {
        UserEntity user = userMapper.selectById(request.getUserId());
        if (ObjectUtils.isEmpty(user)) {
            return 0;
        }
        user.setPassword(passwordEncoder.encode(request.getPassword()));
        return userMapper.updateById(user);
    }

    @Override
    public boolean isManyTeamUser(DeleteRequest deleteRequest) {
        for (int userId : deleteRequest.getIds()) {
            Integer count = teamUserMapper.selectCount(
                    new LambdaQueryWrapper<TeamUserEntity>()
                            .eq(TeamUserEntity::getUserId, userId)
            );
            if (count > 0) {
                return true;
            }
        }
        return false;
    }

    @Override
    public int resetDefaultPassword(Integer id) {
        UserEntity userEntity = userMapper.selectById(id);
        if (ObjectUtils.isEmpty(userEntity)) {
            return 0;
        }
        userEntity.setPassword(passwordEncoder.encode("123456"));
        return userMapper.updateById(userEntity);
    }


}
