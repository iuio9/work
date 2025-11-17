package com.qczy.config;

/**
 * @author ：hh
 * @date ：Created in 2024/7/22 11:17
 * @description：${description}
 * @modified By：
 * @version: $version$
 */


import com.qczy.model.entity.UserEntity;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.ToString;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.GrantedAuthority;

import java.util.Collection;


/**
 * SpringSecurity需要的用户详情
 */



@NoArgsConstructor
@Data
@ToString
public class UserDetails implements org.springframework.security.core.userdetails.UserDetails {

    @Autowired
    private UserEntity user;





    public UserDetails(UserEntity User) {
        this.user = User;
    }





    @Override
    public Collection<? extends GrantedAuthority> getAuthorities() {
     /*   //返回当前用户的权限
        return permissionList.stream()
                .filter(permission -> permission.getValue()!=null)
                .map(permission ->new SimpleGrantedAuthority(permission.getValue()))
                .collect(Collectors.toList());*/
        return null;
    }

    @Override
    public String getPassword() {
        return user.getPassword();
    }

    @Override
    public String getUsername() {
        return user.getUserName();
    }

    @Override
    public boolean isAccountNonExpired() {
        return true;
    }

    @Override
    public boolean isAccountNonLocked() {
        return true;
    }

    @Override
    public boolean isCredentialsNonExpired() {
        return true;
    }

    @Override
    public boolean isEnabled() {
        return true;
    }


}

