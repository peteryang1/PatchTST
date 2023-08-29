#!/bin/bash  
  
# Update package list  
sudo apt update  
  
# Install Zsh  
sudo apt install -y zsh  
  
# Set Zsh as default shell  
chsh -s $(which zsh)  
  
# Install Oh-My-Zsh  
sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"  
  
# Install plugins  
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions  
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting  
git clone https://github.com/rupa/z.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/z  
  
# Add plugins to .zshrc  
sed -i 's/^plugins=(/plugins=(git zsh-autosuggestions z zsh-syntax-highlighting sudo /' ~/.zshrc  
  
# Apply changes  
source ~/.zshrc  

zsh

conda init zsh
  
echo "Zsh, Oh-My-Zsh and plugins have been successfully installed."