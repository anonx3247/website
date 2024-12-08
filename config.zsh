#! /bin/sh

# This script will install a zsh config for anyone running curl anas.lecaillon.com/config.zsh | sh

# Install zsh if not installed
if ! which zsh > /dev/null 2>&1; then
    echo "Installing zsh"
    sudo apt-get install zsh -y
fi

# Install oh-my-zsh
echo "Installing oh-my-zsh"
[[ ! -d ~/.config/oh-my-zsh ]] && curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh | zsh

ZSH="$HOME/.config/oh-my-zsh"
ZSH_CUSTOM="$HOME/.config/oh-my-zsh/custom"
ZSH_THEME="robbyrussell"

# Install zsh-autosuggestions
echo "Installing zsh-autosuggestions"
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions

# Install zsh-syntax-highlighting
echo "Installing zsh-syntax-highlighting"
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

# Install zsh files
echo "Installing zsh files"
cd ~
[[ ! -d ~/.config/zsh ]] && git clone https://github.com/anonx3247/zsh.git ~/.config/zsh
mv ~/.config/zsh/.zshrc ~/.zshrc
mv ~/.config/zsh/.zprofile ~/.zprofile

source ~/.zshrc
echo "Done"